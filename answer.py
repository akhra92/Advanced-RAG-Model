import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from functools import lru_cache
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv(override=True)

MODEL = "openai/gpt-oss-120b"
DB_NAME = str(Path(__file__).parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"

collection_name = "docs"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 2000
RERANK_MAX_TOKENS = 8000
wait = wait_exponential(multiplier=1, min=1, max=20)
stop = stop_after_attempt(3)

CHAT_TIMEOUT = 1200  # seconds; reasoning models can be slow, but never wait forever


def default_token():
    return os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


@lru_cache(maxsize=8)
def get_client(token: str | None = None):
    """
    One client per token, so a deployed app can serve several people who each bring
    their own token and pay for their own calls.
    """
    return InferenceClient(model=MODEL, token=token or default_token(), timeout=CHAT_TIMEOUT)


@lru_cache(maxsize=1)
def get_embedder():
    """Local embeddings: fast, free, no outages. Loaded on first use, not on import."""
    return SentenceTransformer(embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    return PersistentClient(path=DB_NAME).get_or_create_collection(collection_name)


RETRIEVAL_K = 10
FINAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def json_schema_for(model_class):
    """Hugging Face structured output: constrain the reply to this pydantic model's schema"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "schema": model_class.model_json_schema(),
            "strict": True,
        },
    }


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class SearchQuery(BaseModel):
    query: str = Field(
        description="A very short, specific question to search the knowledge base with, e.g. 'Who is Lisa Anderson?'"
    )


def apply_order(chunks, order):
    """Reorder the chunks by the ids the model returned, ignoring anything it made up or dropped"""
    seen = set()
    reordered = []
    for i in order:
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            reordered.append(chunks[i - 1])
    reordered.extend(chunk for i, chunk in enumerate(chunks, 1) if i not in seen)
    return reordered


@retry(wait=wait, stop=stop)
def rerank(question, chunks, token=None):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = get_client(token).chat_completion(
        messages=messages,
        max_tokens=RERANK_MAX_TOKENS,
        response_format=json_schema_for(RankOrder),
    )
    if response.choices[0].finish_reason == "length":
        raise ValueError(
            f"Rerank reply was truncated at {RERANK_MAX_TOKENS} tokens; increase RERANK_MAX_TOKENS or reduce RETRIEVAL_K"
        )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return apply_order(chunks, order)


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop)
def rewrite_query(question, history=[], token=None):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Do NOT answer the question yourself; only provide the short search query.
"""
    response = get_client(token).chat_completion(
        messages=[{"role": "user", "content": message}],
        max_tokens=2000,
        response_format=json_schema_for(SearchQuery),
    )
    reply = response.choices[0].message.content
    return SearchQuery.model_validate_json(reply).query


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def embed(text):
    """Embed a single piece of text locally with sentence-transformers"""
    return get_embedder().encode(text).tolist()


def fetch_context_unranked(question):
    query = embed(question)
    results = get_collection().query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question, token=None):
    chunks = fetch_context_unranked(original_question)
    try:
        rewritten_question = rewrite_query(original_question, token=token)
        chunks = merge_chunks(chunks, fetch_context_unranked(rewritten_question))
    except Exception as e:
        print(f"Query rewrite failed ({type(e).__name__}); using the original question only")
    try:
        chunks = rerank(original_question, chunks, token=token)
    except Exception as e:
        print(f"Rerank failed ({type(e).__name__}); using retrieval order")
    return chunks[:FINAL_K]


def answer_question(
    question: str, history: list[dict] = [], token: str | None = None
) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question, token=token)
    messages = make_rag_messages(question, history, chunks)
    response = get_client(token).chat_completion(messages=messages, max_tokens=MAX_TOKENS)
    return response.choices[0].message.content, chunks
