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
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

DB_PATH = Path(__file__).parent / "preprocessed_db"

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

if not DB_PATH.exists():
    st.error(
        "The knowledge base has not been built yet.\n\n"
        "Ingestion summarises and chunks every document with an LLM, so it is far too "
        "slow to run on first page load. Build it once on your own machine and commit "
        "the result:\n\n"
        "```bash\npython ingest.py\ngit add preprocessed_db && git commit -m "
        '"Add prebuilt vector store"\n```'
    )
    st.stop()

if not token:
    st.info("👈 Enter your Hugging Face access token in the sidebar to start chatting.")
    st.stop()

answer_backend = load_backend()

if answer_backend.get_collection().count() == 0:
    st.error(
        f"`{DB_PATH.name}/` exists but holds no vectors. Rebuild it with `python ingest.py`."
    )
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
