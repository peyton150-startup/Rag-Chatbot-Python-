import streamlit as st
from langchain_community.vectorstores import Chroma

from nvidia_embeddings import NVIDIAOpenAIEmbeddings


@st.cache_resource(show_spinner=False)
def load_vector_store(persist_directory: str) -> Chroma:
    """Open the Chroma store once per Streamlit server.

    Every browser session runs app.py in its own thread. When chromadb upgrades a store
    written by an older version, other threads can query it before the upgrade finishes
    ("no such column: collections.config_json_str"). st.cache_resource makes concurrent
    sessions wait for the first open.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=NVIDIAOpenAIEmbeddings(),
    )
