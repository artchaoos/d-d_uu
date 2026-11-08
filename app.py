"""
D&D 5e Master Rules Companion - Ask-Your-PDFs RAG System
A Streamlit application for querying D&D 5e rulebooks using Groq AI and LangChain (LCEL).
Upload PDFs directly via the web UI. Default model: openai/gpt-oss-20b
"""

import os
import warnings
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="D&D 5e Master Rules Companion",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.1


def extract_documents_from_uploads(uploaded_files) -> List[Document]:
    """
    Extract text from uploaded PDF files using pypdf.
    Create Document objects with strict metadata:
    - filename: original file name (uploaded_file.name)
    - page: 1-indexed page number
    """
    documents = []

    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Read PDF
            pdf_reader = PdfReader(uploaded_file)

            # Extract text page by page
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()

                if text and text.strip():
                    # Create Document with strict metadata
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "filename": uploaded_file.name,
                            "page": page_num + 1  # 1-indexed
                        }
                    )
                    documents.append(doc)

        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


@st.cache_resource(show_spinner="📚 Building vector index from uploaded PDFs...")
def build_vector_store(_documents: List[Document]) -> Chroma:
    """
    Build Chroma vector store from documents.
    Uses _documents prefix to indicate caching by object identity.
    """
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store


def create_rag_chain(vector_store: Chroma, api_key: str, model_name: str):
    """
    Create the RAG chain using modern LCEL (LangChain Expression Language).
    No legacy langchain.chains imports.
    """
    # Initialize Groq LLM with user-selected model
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=1024
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

    # Strict system prompt for D&D 5e Rules DM
    system_prompt = """You are a strict D&D 5e Rules Dungeon Master. Answer the user's question using ONLY the provided context. If the context does not contain the answer, or if you are unsure, you MUST reply exactly with: 'I don't know based on the provided documents.' Do not make up any rules or facts outside the context."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # Format retrieved docs into a single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build LCEL chain: retrieve -> format -> prompt -> llm -> parse
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def format_citations(context_docs) -> List[str]:
    """
    Collect unique source citations from retrieved documents.
    Format: • [filename, p. N]
    """
    citations = set()
    for doc in context_docs:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "?")
        citations.add(f"• [{filename}, p. {page}]")

    return sorted(citations)


def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #5a6c7d;
            margin-bottom: 2rem;
        }
        .response-box {
            background-color: #f8f9fa;
            border-left: 4px solid #1e3a5f;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .citation-box {
            background-color: #eef2f7;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .stSpinner > div {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🎲 D&D 5e Master Rules Companion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your rulebooks — get precise answers with page-level citations</div>', unsafe_allow_html=True)

    # Sidebar for API key and model selection
    with st.sidebar:
        st.header("🔐 Configuration")

        # Groq API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key to enable the LLM. Get one at https://console.groq.com"
        )

        # Model ID input - default to openai/gpt-oss-20b
        groq_model = st.text_input(
            "Model ID",
            value=DEFAULT_GROQ_MODEL,
            placeholder="e.g., openai/gpt-oss-20b, qwen/qwen3-8b-27b, llama-3.3-70b-versatile, etc.",
            help="Enter the Groq model ID. Check https://console.groq.com/docs/models for available models."
        )

        st.divider()

        # File uploader
        st.header("📚 Upload Rulebooks")
        uploaded_files = st.file_uploader(
            "Choose D&D 5e PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more D&D 5e rulebook PDFs (Basic Rules, PHB, DMG, etc.)"
        )

        st.divider()

        # Status indicators
        st.subheader("📊 System Status")

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            for f in uploaded_files:
                st.caption(f"📄 {f.name}")
        else:
            st.info("📤 No files uploaded yet")

        st.caption(f"Model: `{groq_model}` | Temperature: `{TEMPERATURE}` | Top-K: `{RETRIEVER_K}`")
        st.caption(f"Embeddings: `{EMBEDDING_MODEL}` | Chunk: `{CHUNK_SIZE}`/{CHUNK_OVERLAP}")

    # Main content area
    if not uploaded_files:
        st.markdown("""
            <div class="info-box">
                <h4>👋 Welcome!</h4>
                <p>Upload your D&D 5e rulebook PDFs using the sidebar to get started.</p>
                <p><strong>Supported files:</strong> Any PDF (Basic Rules, Player's Handbook, Dungeon Master's Guide, etc.)</p>
                <p><strong>How it works:</strong></p>
                <ol>
                    <li>Upload PDFs via the sidebar</li>
                    <li>System extracts text and builds a searchable index</li>
                    <li>Ask rules questions — get answers with page citations</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        return

    if not groq_api_key:
        st.markdown("""
            <div class="warning-box">
                <h4>🔑 API Key Required</h4>
                <p>Enter your Groq API key in the sidebar to start querying the rulebooks.</p>
                <p>Get a free API key at <a href="https://console.groq.com" target="_blank">console.groq.com</a></p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Process uploaded files -> build vector store
    with st.spinner("📖 Processing uploaded PDFs..."):
        try:
            # Extract documents from uploads
            raw_documents = extract_documents_from_uploads(uploaded_files)

            if not raw_documents:
                st.error("No readable text found in uploaded PDFs. Please check your files.")
                return

            # Split into chunks
            chunks = split_documents(raw_documents)

            # Build vector store (cached)
            vector_store = build_vector_store(chunks)

            st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)")

        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return

    # Query input
    st.subheader("🔍 Ask the Rules")
    user_query = st.text_area(
        "Your question:",
        placeholder="e.g., How does the Shield spell work? What are the rules for grappling? What is the DC for a very hard check?",
        height=100,
        help="Ask any D&D 5e rules question. The system will search the indexed PDFs and provide answers with citations."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Process query
    if ask_button and user_query.strip():
        with st.spinner("🧠 Consulting the rulebooks..."):
            try:
                # Create RAG chain (LCEL) with user-selected model
                rag_chain, retriever = create_rag_chain(vector_store, groq_api_key, groq_model)

                # Retrieve context docs for citations (separate from chain for display)
                context_docs = retriever.invoke(user_query.strip())

                # Invoke chain with question
                answer = rag_chain.invoke({"question": user_query.strip()})

                # Display answer
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown("### 📖 Answer")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display citations
                if context_docs:
                    citations = format_citations(context_docs)
                    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                    st.markdown("### 📎 Sources")
                    for citation in citations:
                        st.markdown(citation)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No source documents were retrieved for this answer.")

            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("Please check your API key and model ID, then try again.")

    elif ask_button and not user_query.strip():
        st.warning("Please enter a question before asking.")


if __name__ == "__main__":
    main()