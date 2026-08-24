# Advanced RAG Model

An advanced retrieval-augmented chat assistant that answers questions about Insurellm from a local markdown knowledge base.

[Live Demo](https://share.streamlit.app/) — you need to bring your own OpenAI API key.

## Stack

- Streamlit — chat UI
- LangChain + Chroma — retrieval
- OpenAI `gpt-oss-120b` (chat) and `all-MiniLM-L6-v2` (embeddings)

## Setup

```bash
pip install -r requirements.txt
```

The app asks for an OpenAI API key in the sidebar. For local runs you can skip that prompt by putting the key in a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
streamlit run app.py
```

On the first run the app builds the vector store from `knowledge-base/` into `vector_db/`. Later runs reuse it. To rebuild manually:

```bash
python ingest.py
```

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit chat interface |
| `answer.py` | Retrieval and answer generation |
| `ingest.py` | Loads `knowledge-base/`, chunks it, writes embeddings to `vector_db/` |
| `knowledge-base/` | Source markdown documents |
