import base64
import os
from pathlib import Path

import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from nvidia_chat import create_nvidia_chat_model
from vector_store import load_vector_store


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Harmony Aesthetics & Wellness",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS – light purple / spa aesthetic
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #824A9C;
        color: #ffffff;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    /* Remove Streamlit default blue highlights */
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Header bar */
    header[data-testid="stHeader"] {
        background: #824A9C;
    }

    /* Page text and labels */
    .stApp p, .stApp label {
        color: #ffffff;
    }

    /* Titles */
    h1 {
        color: #ffffff;
        font-weight: 500;
        letter-spacing: 0.4px;
    }

    h2, h3, h5 {
        color: #f3e8f7;
        font-weight: 400;
    }

    /* Text input */
    div[data-baseweb="input"] > div {
        background-color: #ffffff;
        border: 1px solid #d8d2df;
        border-radius: 6px;
    }

    div[data-baseweb="input"] input {
        color: #2b2b2b;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #b9a7c8;
        box-shadow: 0 0 0 1px #b9a7c8;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #b9a7c8;
        color: #ffffff;
        border: none;
        border-radius: 6px;
    }

    button[kind="primary"]:hover {
        background-color: #a996ba;
    }

    /* Answer container */
    .answer-box {
        background-color: #ffffff;
        border-left: 4px solid #b9a7c8;
        padding: 1.2em;
        border-radius: 6px;
        margin-top: 1em;
        font-size: 1rem;
    }

    .answer-box, .answer-box * {
        color: #2b2b2b;
    }

    /* Expander */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #ffffff;
    }

    /* Sources */
    .source-text {
        font-size: 0.85rem;
        color: #f3e8f7;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Logo + Header
# --------------------------------------------------
logo = base64.b64encode(Path(__file__).with_name("logo.svg").read_bytes()).decode()
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:1rem;">
        <img
            src="data:image/svg+xml;base64,{logo}"
            style="max-width:180px;"
        />
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Harmony Aesthetics & Wellness")
st.markdown(
    "##### Personalized, clinically grounded answers to your aesthetic and wellness questions."
)

# --------------------------------------------------
# LangChain setup
# --------------------------------------------------
if not os.getenv("NVIDIA_API_KEY"):
    st.error("Set NVIDIA_API_KEY before starting the app.")
    st.stop()

persist_directory = os.getenv("CHROMA_DB_DIR", "db_nvidia")
if not os.path.isdir(persist_directory):
    st.error(f"Vector database not found at '{persist_directory}'. Run `python ingest.py` first.")
    st.stop()

llm = create_nvidia_chat_model()
db = load_vector_store(persist_directory)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# --------------------------------------------------
# Input + Output
# --------------------------------------------------
query = st.text_input("Ask a question about treatments, services, or policies")

if query:
    with st.spinner("Preparing your response…"):
        result = qa.invoke({"question": query})

    st.markdown(
        f"<div class='answer-box'>{result['answer']}</div>",
        unsafe_allow_html=True
    )

    with st.expander("Source documents"):
        for doc in result["source_documents"]:
            st.markdown(
                f"<div class='source-text'>• {doc.metadata.get('source', 'unknown')}</div>",
                unsafe_allow_html=True
            )
