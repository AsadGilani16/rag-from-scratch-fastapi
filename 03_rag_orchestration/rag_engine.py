import chromadb
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "02_vector_database", "my_chroma_db")
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="pdf_documents")


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


def semantic_search(query: str, top_k: int = 3) -> list[str]:

    chroma_client = chromadb.PersistentClient(path="../02_vector_database/my_chroma_db")

    collection = chroma_client.get_or_create_collection(name="pdf_documents")

    results = collection.query(query_texts=[query], n_results=top_k)

    retrieved_chunks = results["documents"][0]

    return retrieved_chunks

def generate_rag_answer(query: str, context_text: str) -> str:
    system_instruction = (
        "You are an assistant for answering questions based strictly on provided context. "
        "If you do not know the answer based on the context, say 'I cannot find the answer in the provided documents.' "
        "Do not use outside knowledge."
    )

    user_prompt = f"""Context:
{context_text}

Question: {query}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    
    ingest_document("sample.pdf") 

    ##print("\n--- DB Verification ---")
    ##print("Total chunks stored:", collection.count())
    ##print("Sample retrieved chunk:", collection.peek(limit=1))

    #sample_query = "What are the main diplomatic and economic challenges?"
    #chunks = semantic_search(sample_query, top_k=3)

    #print(f"Retrieved {len(chunks)} chunks for query: '{sample_query}'\n")
    #for i, chunk in enumerate(chunks, 1):
        #print(f"--- Chunk {i} ---")
        #print(chunk)
        #print()

    user_query = "What are the main diplomatic and economic challenges?"
    chunks = semantic_search(user_query, top_k=3)
    context = "\n\n---\n\n".join(chunks) #formatting

    answer = generate_rag_answer(user_query, context)

    # C. Output
    print(f"\nQuestion: {user_query}\n")
    print(f"RAG Answer:\n{answer}")

