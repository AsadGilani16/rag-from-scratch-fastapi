import os
import chromadb
from pypdf import PdfReader

client = chromadb.PersistentClient(path="./my_chroma_db")
collection = client.get_or_create_collection(name="pdf_knowledge_base")

def extract_text(file_path: str) -> str:
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks

def ingest_document(file_path: str):
    print(f"Reading {file_path}...")
    raw_text = extract_text(file_path)
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    
    file_name = os.path.basename(file_path)
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"Successfully ingested {len(chunks)} chunks from '{file_name}' into ChromaDB!")

if __name__ == "__main__":
    
    ingest_document("sample.pdf")

    print("\n--- DB Verification ---")
    print("Total chunks stored:", collection.count())
    print("Sample retrieved chunk:", collection.peek(limit=1))