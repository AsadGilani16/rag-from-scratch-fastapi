import os
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "02_vector_database", "my_chroma_db")
)

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="pdf_documents")

DISTANCE_THRESHOLD = 1.2  

def retrieve_context(query: str, top_k: int = 3):
    results = collection.query(
        query_texts=[query],
          n_results=top_k, 
          include=["documents", "distances", "metadatas"]
    )

    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not distances or distances[0] > DISTANCE_THRESHOLD:
        return None, None

    return documents, metadatas

def evaluate_groundedness(context: str, answer: str) -> bool:
    eval_prompt = f"""
    Context:
    {context}

    Answer to Evaluate:
    {answer}

    Task: Is the provided Answer strictly and entirely supported by the Context above? 
    Do not use outside knowledge. Answer with ONLY the single word 'YES' or 'NO'.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,
    )

    eval_result = response.choices[0].message.content.strip().upper()
    return "YES" in eval_result

def run_rag_pipeline(query: str, top_k: int = 3):
    print(f"\nQuery: {query}")

    # Pass top_k to your retrieval function
    documents, metadatas = retrieve_context(query, top_k=top_k)

    # 1. Fallback if no relevant documents were retrieved
    if not documents:
        return {
            "query": query,
            "answer": "Information not found in context.",
            "is_grounded": False,
            "sources": []
        }

    context_blocks = []
    sources_list = []

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        
        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc}")
        
        # Collect structured source info for Pydantic response
        sources_list.append({"source": source, "page": page})

    full_context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the information is insufficient, say 'Information not found in context.' "
        "Always cite your sources using the source tags provided in the context."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{full_context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    # Run post-generation evaluation check
    is_grounded = evaluate_groundedness(full_context, answer)

    print(f"\nGenerated Answer:\n{answer}")
    print(f"\n[Eval Check] Grounded in Context: {is_grounded}")

    # 2. Return full dictionary matching QueryResponse schema
    return {
        "query": query,
        "answer": answer,
        "is_grounded": is_grounded,
        "sources": sources_list
    }


def main():
    print("=" * 60)
    print("  RAG Engine with Evaluation & Anti-Hallucination Guardrails")
    print("=" * 60)

    # Test 1: In-domain question
    query_1 = "According to the text, what specific events led to the formation of the Tehrik-i-Taliban Pakistan (TTP) in 2007?"
    print(f"\n--- [Test 1: Valid Query] ---")
    print(f"Query: {query_1}")
    
    # Store the returned answer and print it!
    result_1 = run_rag_pipeline(query_1)
    print(f"\nResponse:\n{result_1}")

    print("\n" + "-" * 60)

    # Test 2: Out-of-domain question
    query_2 = "What is the capital of France and its population?"
    print(f"\n--- [Test 2: Out-of-Domain Query] ---")
    print(f"Query: {query_2}")
    
    # Store the returned answer and print it!
    result_2 = run_rag_pipeline(query_2)
    print(f"\nResponse:\n{result_2}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()