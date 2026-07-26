import streamlit as st

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


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
        background-color: #f8f6f9;
        color: #2b2b2b;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    /* Remove Streamlit default blue highlights */
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Header bar */
    header {
        background: #f8f6f9;
    }

    /* Titles */
    h1 {
        color: #4b4453;
        font-weight: 500;
        letter-spacing: 0.4px;
    }

    h2, h3 {
        color: #6a6272;
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

    /* Expander */
    .stExpanderHeader {
        color: #6a6272;
    }

    /* Sources */
    .source-text {
        font-size: 0.85rem;
        color: #6f6a75;
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
st.markdown(
    """
    <div style="text-align:center; margin-bottom:1rem;">
        <img
            src="https://harmonyaestheticswellness.com/wp-content/uploads/2023/03/cropped-HAW_Logo_PMS_BLUE.png"
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
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

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
