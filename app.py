"""Streamlit chat UI for the Advanced RAG assistant, ready for Streamlit Community Cloud."""

import sys

# Some Streamlit Cloud images ship an sqlite3 older than chromadb requires; if the
# pysqlite3-binary wheel is installed, swap it in before chromadb is ever imported.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import threading

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="Insurellm Expert Assistant", page_icon="🏢", layout="wide")


def from_secrets(name: str) -> str:
    """Read a Streamlit secret, tolerating the missing secrets.toml of a local run."""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


@st.cache_resource(show_spinner="Warming up the retriever, this runs only once...")
def load_backend():
    """
    Import the RAG backend and pay the one-off costs up front: downloading the local
    embedding model and opening the Chroma store. Cached for the life of the deployment.
    """
    import answer

    answer.get_embedder()
    answer.get_collection()
    return answer


@st.cache_resource
def build_lock():
    """One lock per deployment, so two arrivals can't kick off ingestion at the same time."""
    return threading.Lock()


def indexed_chunks(answer) -> int:
    try:
        return answer.get_collection().count()
    except Exception:
        return 0


def build_knowledge_base(answer, token):
    """
    Index the knowledge base on first use, so the app runs from a clean checkout with
    no preprocessed_db/. Ingestion is one LLM call per document, so this is slow; it
    runs once per deployment, for whoever happens to arrive first.
    """
    if indexed_chunks(answer):
        return

    with build_lock():
        if indexed_chunks(answer):  # another session may have built it while we waited
            return

        import ingest

        with st.status(
            "Building the knowledge base, this runs only once...", expanded=True
        ) as status:
            st.write("Loading the documents...")
            documents = ingest.fetch_documents()
            bar = st.progress(0.0, text=f"Chunking 0 / {len(documents)} documents")

            def on_progress(done, total):
                bar.progress(done / total, text=f"Chunking {done} / {total} documents")

            chunks = ingest.create_chunks(documents, token=token, on_progress=on_progress)
            st.write(f"Embedding {len(chunks)} chunks...")
            ingest.create_embeddings(chunks)
            answer.get_collection.cache_clear()  # ingestion replaced the collection
            status.update(
                label=f"Knowledge base ready — {indexed_chunks(answer):,} chunks indexed",
                state="complete",
                expanded=False,
            )


with st.sidebar:
    st.header("Insurellm Expert Assistant")
    st.write(
        "A retrieval-augmented chat assistant over the Insurellm knowledge base, "
        "with query rewriting and LLM re-ranking."
    )

    st.subheader("🔑 Your Hugging Face token")
    typed_token = st.text_input(
        "Hugging Face access token",
        type="password",
        placeholder="hf_...",
        label_visibility="collapsed",
        help="Used only for your own requests and never stored.",
    ).strip()
    st.caption(
        "Get one at [huggingface.co/settings/tokens]"
        "(https://huggingface.co/settings/tokens) with the "
        "*Make calls to Inference Providers* permission. "
        "It lives in your browser session only and is gone when you close the tab."
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = []
        st.rerun()

# The deployer may supply a shared token instead; the user's own always wins.
token = (
    typed_token
    or from_secrets("HF_TOKEN")
    or from_secrets("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN", "")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []

st.title("🏢 Insurellm Expert Assistant")
st.caption("Ask me anything about Insurellm!")

if not token:
    st.info("👈 Enter your Hugging Face access token in the sidebar to start chatting.")
    st.stop()

answer_backend = load_backend()

if not indexed_chunks(answer_backend):
    document_count = len(list(answer_backend.KNOWLEDGE_BASE_PATH.rglob("*.md")))
    st.warning(
        "The knowledge base has not been indexed yet. Building it summarises and chunks "
        f"all {document_count} documents with an LLM, which takes several minutes and uses "
        "your inference quota. To skip the wait next time, run `python ingest.py` locally "
        "and commit `preprocessed_db/`."
    )

try:
    build_knowledge_base(answer_backend, token)
except Exception as error:
    st.error(f"Could not build the knowledge base — check that your token is valid.\n\n{error}")
    st.stop()

chat_column, context_column = st.columns(2)

with chat_column:
    st.subheader("💬 Conversation")
    transcript = st.container(height=600)
    for message in st.session_state.messages:
        transcript.chat_message(message["role"]).markdown(message["content"])

question = st.chat_input("Ask anything about Insurellm...")

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    transcript.chat_message("user").markdown(question)

    with transcript.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, re-ranking and thinking..."):
                reply, context = answer_backend.answer_question(question, history, token=token)
        except Exception as error:
            st.session_state.messages.pop()
            st.error(
                "Request failed — check that your token is valid and has Inference "
                f"Providers access.\n\n{error}"
            )
            st.stop()
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.context = context

with context_column:
    st.subheader("📚 Retrieved Context")
    panel = st.container(height=600)
    if st.session_state.context:
        for doc in st.session_state.context:
            panel.markdown(f":orange[**Source:** {doc.metadata.get('source', 'unknown')}]")
            panel.markdown(doc.page_content)
            panel.divider()
    else:
        panel.markdown("*Retrieved context will appear here*")
