# Advanced RAG Model

An advanced retrieval-augmented chat assistant that answers questions about Insurellm from a
local markdown knowledge base. Every answer goes through query rewriting, dense retrieval and
an LLM re-ranking pass before generation.

[Live Demo](https://share.streamlit.app/) — you need to bring your own Hugging Face token.

## Stack

- Streamlit — chat UI
- Chroma — persistent vector store
- `sentence-transformers/all-MiniLM-L6-v2` — embeddings, computed locally, no API key needed
- `openai/gpt-oss-120b` via Hugging Face Inference Providers — chunking, rewriting, re-ranking, answering

## Setup

```bash
pip install -r requirements.txt
```

The app asks for a Hugging Face token in the sidebar. For local runs you can skip that prompt by
putting the token in a `.env` file in the project root:

```
HF_TOKEN=hf_...
```

The token needs the **Make calls to Inference Providers** permission.

## Build the knowledge base

Ingestion asks the LLM to split each document into overlapping, summarised chunks, so it takes
several minutes over the 76 documents — far too slow to run on a page load. Build it once
locally and commit the result, so the deployed app only ever reads it:

```bash
python ingest.py
git add preprocessed_db && git commit -m "Add prebuilt vector store"
```

## Run

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push the repo — including `preprocessed_db/` — to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. Optionally add a shared token under **Settings → Secrets** so visitors don't need their own:

   ```toml
   HF_TOKEN = "hf_..."
   ```

   Leave it out and each visitor supplies their own token in the sidebar.

If the build fails on `sqlite3` being too old for Chroma, add `pysqlite3-binary` to
`requirements.txt`; `app.py` already swaps it in when it is present.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit chat interface |
| `answer.py` | Retrieval, query rewriting, re-ranking and answer generation |
| `ingest.py` | Loads `knowledge-base/`, LLM-chunks it, writes embeddings to `preprocessed_db/` |
| `knowledge-base/` | Source markdown documents |
