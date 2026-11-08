# 🎲 D&D 5e Master Rules Companion

A Streamlit-based RAG (Retrieval-Augmented Generation) application that allows Dungeon Masters to upload D&D 5e rulebooks in PDF format directly through the web UI and receive accurate answers with precise page-level citations.

## Features

- **Web-Based PDF Upload**: Drag-and-drop multiple PDFs via `st.file_uploader` — no local folder setup required
- **Groq AI Integration**: Uses `ChatGroq` with `openai/gpt-oss-20b` (configurable) for fast, deterministic responses
- **Page-Level Citations**: Every answer includes unique `• [filename, p. N]` citations
- **Strict Anti-Hallucination**: Enforces "I don't know based on the provided documents" when context is insufficient
- **Modern LCEL Architecture**: LangChain Expression Language with `RunnablePassthrough` — no legacy chains
- **Cached Indexing**: Vector store built once per session via `@st.cache_resource`
- **Clean UI**: Wide-mode layout with sidebar configuration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### 3. Configure & Use

1. **Enter Groq API Key** in the sidebar (get one at [console.groq.com](https://console.groq.com))
2. **Verify Model ID** — default is `openai/gpt-oss-20b`
3. **Upload PDFs** — drag and drop D&D rulebooks (Basic Rules, PHB, DMG, etc.) in the sidebar
4. **Ask Questions** — type any D&D 5e rules question and get cited answers

```
project_root/
├── app.py
├── requirements.txt
└── .env.example (optional)
```

## Architecture

### Ingestion Pipeline (Web-Uploaded)
1. **Upload**: `st.file_uploader` accepts multiple PDFs via browser
2. **Extract**: `pypdf.PdfReader` reads each file page-by-page from memory
3. **Document Creation**: Each page → `Document(page_content, metadata={filename, page})`
   - `filename`: Original uploaded file name
   - `page`: 1-indexed page number
4. **Split**: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
5. **Embed**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
6. **Store**: Local Chroma vector database (`./chroma_db/`)

### RAG Chain (LCEL)
```
User Question
      │
      ▼
RunnablePassthrough.assign(context=retriever.invoke(question))
      │
      ▼
ChatPromptTemplate (strict system prompt)
      │
      ▼
ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
      │
      ▼
StrOutputParser()
      │
      ▼
Answer + Citations
```

- **Retriever**: Top-K=4 similarity search
- **Prompt**: Strict DM persona — context-only, exact fallback string
- **Output**: Answer + unique `• [filename, p. N]` citations

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHUNK_SIZE` | 1000 | Text chunk size in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVER_K` | 4 | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | HuggingFace embedding model |
| `DEFAULT_GROQ_MODEL` | openai/gpt-oss-20b | Default Groq model ID |
| `TEMPERATURE` | 0.1 | LLM temperature (deterministic) |

All settings adjustable in `app.py` constants; Model ID overridable in sidebar.

## Usage Tips

- **Specific questions work best**: "What is the DC for a very hard ability check?" vs "Tell me about checks"
- **Citations are precise**: `• [Basic_Rules.pdf, p. 72]` — verify in your PDF
- **First upload indexes PDFs**: Subsequent questions are instant (cached)
- **Temperature 0.1**: Ensures consistent, rule-accurate responses
- **Change models anytime**: Enter any valid Groq model ID in sidebar

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No files uploaded" | Use sidebar uploader — drag PDFs into the widget |
| API key errors | Verify key at [console.groq.com](https://console.groq.com) |
| Model not found | Check [Groq Models](https://console.groq.com/docs/models) for current IDs |
| Slow first query | Normal — indexing runs on first upload |
| Chroma errors | Delete `./chroma_db/` and re-upload |

## Tech Stack

- **Frontend**: Streamlit 1.32+
- **LLM**: Groq AI via `langchain-groq` (`ChatGroq`)
- **Model**: `openai/gpt-oss-20b` (configurable)
- **Framework**: LangChain 0.3+ with LCEL (`RunnablePassthrough`, `StrOutputParser`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`
- **Vector Store**: Chroma via `langchain-chroma`
- **PDF Parsing**: `pypdf` (page-level extraction)
- **Text Splitting**: `RecursiveCharacterTextSplitter`
