# Custom RAG Pipeline from Scratch

A lightweight, zero-framework Retrieval-Augmented Generation (RAG) pipeline built with Python, ChromaDB, and Groq LLMs. This repository demonstrates the core mechanics of RAG — from raw text embeddings and vector store ingestion to context injection and LLM orchestration — without relying on heavy abstraction wrappers like LangChain or LlamaIndex.

---

## 🛠️ Tech Stack & Key Tools

| Component | Tool |
|---|---|
| **Language** | Python 3.11+ |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (Persistent Local Storage) |
| **LLM Engine** | [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`) |
| **Document Parser** | `pypdf` |
| **Environment Management** | `python-dotenv` |

---

## 📁 Repository Structure

```text
rag-from-scratch-fastapi/
│
├── 01_embeddings_and_similarity/  # Fundamentals of vector embeddings and cosine math
├── 02_vector_database/            # PDF ingestion, chunking strategy, & ChromaDB persistence
├── 03_rag_orchestration/          # Custom RAG engine, context injection, & Groq API execution
├── .gitignore                     # Environment variables & DB exclusions
├── requirements.txt               # Core dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Key Features Built So Far

### 1. Vector Fundamentals & Embeddings (`01_embeddings_and_similarity`)
- Exploration of text embedding spaces and vector representations.
- Implementation of raw similarity search mechanics using cosine distance.

### 2. PDF Ingestion & Persistence (`02_vector_database`)
- PDF document parsing using `pypdf`.
- Text chunking strategy for long-context documents.
- Persistent vector database storage using ChromaDB to eliminate redundant re-ingestion.

### 3. RAG Orchestration Engine (`03_rag_orchestration`)
- Semantic search function retrieving top-*k* relevant chunks dynamically.
- Custom prompt engineering with strict system boundaries to prevent hallucinations.
- LLM response generation via low-latency Groq client (`llama-3.3-70b-versatile`).

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

```bash
# Step 1: Ingest and chunk a PDF into the vector store
python 02_vector_database/ingest.py --file path/to/document.pdf

# Step 2: Run the RAG pipeline and ask questions
python 03_rag_orchestration/main.py --query "What is this document about?"
```

*(Adjust script names/paths above to match your actual filenames.)*

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
                                                          │  Prompt Construction │
                                                          │  (Context Injection) │
                                                          └──────────────────────┘
                                                                        │
                                                                        ▼
                                                          ┌──────────────────────┐
                                                          │   Groq LLM Response  │
                                                          │ (llama-3.3-70b)      │
                                                          └──────────────────────┘
```

1. **Ingest** — PDFs are parsed and split into overlapping text chunks.
2. **Embed** — Each chunk is converted into a vector embedding.
3. **Store** — Embeddings are persisted locally in ChromaDB, avoiding re-ingestion on every run.
4. **Retrieve** — A user query is embedded and matched against stored vectors via cosine similarity to fetch the top-*k* most relevant chunks.
5. **Generate** — Retrieved chunks are injected into a tightly-scoped system prompt, and the Groq-hosted LLM generates a grounded, hallucination-resistant answer.

---

## 🗺️ Roadmap

- [ ] Add support for additional document formats (DOCX, TXT, Markdown)
- [ ] Implement re-ranking of retrieved chunks
- [ ] Add a simple FastAPI wrapper for HTTP-based querying
- [ ] Add evaluation scripts for retrieval quality
- [ ] Dockerize the pipeline for easier deployment

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
