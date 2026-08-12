# Custom RAG Pipeline from Scratch

A lightweight, zero-framework Retrieval-Augmented Generation (RAG) pipeline built with Python, ChromaDB, and Groq LLMs — served over a FastAPI HTTP API. This repository demonstrates the core mechanics of RAG — from raw text embeddings and vector store ingestion to context injection, groundedness evaluation, and LLM orchestration — without relying on heavy abstraction wrappers like LangChain or LlamaIndex.

---

## 🛠️ Tech Stack & Key Tools

| Component | Tool |
|---|---|
| **Language** | Python 3.11+ |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (Persistent Local Storage) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **LLM Engine** | [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant` for eval) |
| **Document Parser** | `pypdf` |
| **Environment Management** | `python-dotenv` |
| **Containerization** | Docker |

---

## 📁 Repository Structure

```text
rag-from-scratch-fastapi/
│
├── 01_embeddings_and_similarity/  # Fundamentals of vector embeddings and cosine similarity
│   ├── basic_embedding_test.py
│   └── simple_rag_retrieval.py
│
├── 02_vector_database/            # PDF ingestion, chunking strategy, & ChromaDB persistence
│   ├── chroma_basics.py
│   ├── ingest.py
│   └── sample.pdf
│
├── 03_rag_orchestration/          # RAG engine, FastAPI service, and evaluation guardrails
│   ├── main.py                    # FastAPI app: /upload and /ask endpoints
│   ├── rag_engine.py              # Ingestion + semantic search + generation
│   ├── rag_eval_engine.py         # Retrieval, distance-threshold filtering, groundedness check
│   ├── schemas.py                 # Pydantic request/response models
│   └── sample.pdf
│
├── Dockerfile                     # Containerized FastAPI service
├── .dockerignore
├── .gitignore                     # Environment variables & DB exclusions
├── requirements.txt                # Core dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Key Features

### 1. Vector Fundamentals & Embeddings (`01_embeddings_and_similarity`)
- Exploration of text embedding spaces and vector representations with `sentence-transformers`.
- Raw cosine-similarity search mechanics on toy sentence sets.

### 2. PDF Ingestion & Persistence (`02_vector_database`)
- PDF and TXT document parsing using `pypdf`.
- Overlapping text chunking strategy for long-context documents.
- Persistent vector database storage using ChromaDB to avoid redundant re-ingestion.

### 3. RAG Orchestration Engine (`03_rag_orchestration`)
- Semantic search retrieving top-*k* relevant chunks dynamically.
- Custom prompt engineering with strict system boundaries to reduce hallucinations.
- LLM response generation via the low-latency Groq API (`llama-3.3-70b-versatile`).

### 4. Evaluation & Guardrails (`rag_eval_engine.py`)
- Distance-threshold filtering: queries with no sufficiently relevant chunks are rejected before hitting the LLM.
- Automated groundedness check — a second, faster LLM call (`llama-3.1-8b-instant`) verifies whether the generated answer is actually supported by the retrieved context.
- Source attribution (document + page) returned alongside every answer.

### 5. FastAPI Web Service (`main.py`)
- `POST /upload` — upload a `.pdf` or `.txt` file directly; it's chunked and ingested into ChromaDB on the fly.
- `POST /ask` — ask a question and get back an answer, a groundedness flag, and the source chunks used.
- `GET /` — health check.

### 6. Dockerized Deployment
- A `Dockerfile` builds and serves the FastAPI app with Uvicorn, ready to run in a container.

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.11 or higher
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/rag-from-scratch-fastapi.git
cd rag-from-scratch-fastapi

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Usage

**Option A — Run the API server:**

```bash
uvicorn 03_rag_orchestration.main:app --reload --port 8000
```

Then interact with it (e.g. via the auto-generated docs at `http://localhost:8000/docs`):

```bash
# Upload a document
curl -X POST http://localhost:8000/upload -F "file=@path/to/document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 3}'
```

**Option B — Run the pipeline directly from the command line:**

```bash
# Step 1: Ingest and chunk a PDF into the vector store
python 02_vector_database/ingest.py

# Step 2: Run the RAG + evaluation pipeline
python 03_rag_orchestration/rag_eval_engine.py
```

**Option C — Run with Docker:**

```bash
docker build -t rag-from-scratch .
docker run -p 8000:8000 --env-file .env rag-from-scratch
```

---

## 🧠 How It Works

```text
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│  Raw PDF    │ --> │  Chunking &  │ --> │   ChromaDB      │ --> │  Top-k      │
│  Document   │     │  Embedding   │     │   (Persistent)  │     │  Retrieval  │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │  Distance Threshold  │
                                                          │   Relevance Filter   │
                                                          └──────────────────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │  Prompt Construction │
                                                          │  (Context Injection) │
                                                          └──────────────────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │   Groq LLM Response  │
                                                          │ (llama-3.3-70b)      │
                                                          └──────────────────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │ Groundedness Check   │
                                                          │ (llama-3.1-8b)       │
                                                          └──────────────────────┘
```

1. **Ingest** — PDFs/TXT files are parsed and split into overlapping text chunks (via `/upload` or the CLI script).
2. **Embed** — Each chunk is converted into a vector embedding.
3. **Store** — Embeddings are persisted locally in ChromaDB, avoiding re-ingestion on every run.
4. **Retrieve** — A user query is embedded and matched against stored vectors via cosine distance to fetch the top-*k* most relevant chunks.
5. **Filter** — If the closest match exceeds the distance threshold, the query is rejected as out-of-scope before reaching the LLM.
6. **Generate** — Retrieved chunks are injected into a tightly-scoped system prompt, and the Groq-hosted LLM generates a grounded answer with source citations.
7. **Evaluate** — A second, cheaper LLM call checks whether the answer is actually supported by the retrieved context, returned as an `is_grounded` flag.

---

## 🗺️ Roadmap

- [ ] Add support for additional document formats (DOCX, Markdown)
- [ ] Implement re-ranking of retrieved chunks
- [ ] Add automated test suite for retrieval and evaluation quality
- [ ] Add per-document delete/list endpoints
- [ ] Streaming responses from `/ask`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for the lightweight vector store
- [Groq](https://groq.com/) for blazing-fast LLM inference
- [pypdf](https://pypi.org/project/pypdf/) for PDF parsing
- [FastAPI](https://fastapi.tiangolo.com/) for the web layer
