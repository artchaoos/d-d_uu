# 🎲 D&D 5e Master Rules Companion

A Streamlit-based RAG (Retrieval-Augmented Generation) application that allows Dungeon Masters to query D&D 5e rulebooks in PDF format and receive accurate answers with precise page-level citations.

## Features

- **PDF-based Knowledge Base**: Load D&D 5e rulebooks from a local `sample_books/` directory
- **Groq AI Integration**: Uses `ChatGroq` with `llama3-8b-8192` for fast, deterministic responses
- **Page-level Citations**: Every answer includes precise `[filename, p. N]` citations
- **Strict Anti-Hallucination**: Enforces "I don't know based on the provided documents" when context is insufficient
- **Cached Indexing**: Vector store initialization cached with `@st.cache_resource` for fast subsequent runs
- **Clean UI**: Wide-mode layout with sidebar configuration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Rulebooks

Place your D&D 5e PDF files in the `sample_books/` directory:

```
project_root/
├── app.py
├── requirements.txt
└── sample_books/
    ├── Basic_Rules.pdf
    ├── Players_Handbook.pdf
    └── Dungeon_Masters_Guide.pdf
```

### 3. Configure API Key

**Option A: Sidebar Input (Recommended)**
- Run the app and enter your Groq API key in the sidebar

**Option B: Environment Variable**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free API key at: https://console.groq.com

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Architecture

### Ingestion Pipeline
1. **Loader**: `PyPDFDirectoryLoader` reads all PDFs from `sample_books/`
2. **Splitter**: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
3. **Metadata Formatting**: 
   - `filename`: Base filename only (e.g., "Basic_Rules.pdf")
   - `page`: 1-indexed page numbers (PyPDF is 0-indexed)
4. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
5. **Vector Store**: Local Chroma database (`./chroma_db/`)

### RAG Chain
1. **Retriever**: Top-K (k=4) similarity search
2. **LLM**: `ChatGroq` with `llama3-8b-8192`, temperature=0.1
3. **Prompt**: Strict system prompt enforcing context-only answers
4. **Chain**: Modern LangChain `create_stuff_documents_chain` + `create_retrieval_chain`

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 1000 | Text chunk size in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVER_K` | 4 | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | HuggingFace embedding model |
| `GROQ_MODEL` | llama3-8b-8192 | Groq model ID |
| `TEMPERATURE` | 0.1 | LLM temperature (deterministic) |

## Usage Tips

- **Specific questions work best**: "What is the DC for a very hard ability check?" vs "Tell me about checks"
- **Citations are clickable references**: Use them to verify answers in your physical/digital rulebooks
- **First run indexes PDFs**: Subsequent runs are instant thanks to caching
- **Temperature 0.1**: Ensures consistent, rule-accurate responses

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No PDFs found" | Ensure PDFs are in `sample_books/` directory |
| API key errors | Verify key at https://console.groq.com |
| Slow first run | Normal - indexing takes 30-60 seconds |
| Chroma errors | Delete `./chroma_db/` and restart |

## License

Educational use - College Assignment