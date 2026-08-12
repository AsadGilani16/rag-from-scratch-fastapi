import os
import asyncio
import chromadb
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
groq_client = AsyncGroq()

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

async def evaluate_groundedness(context: str, answer: str) -> bool:
    eval_prompt = f"""
    Context:
    {context}

    Answer to Evaluate:
    {answer}

    Task: Is the provided Answer strictly and entirely supported by the Context above? 
    Do not use outside knowledge. Answer with ONLY the single word 'YES' or 'NO'.
    """

    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0,
    )

    eval_result = response.choices[0].message.content.strip().upper()
    return "YES" in eval_result

async def run_rag_pipeline(query: str, top_k: int = 3):
    print(f"\nQuery: {query}")

    documents, metadatas = retrieve_context(query, top_k=top_k)

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
        sources_list.append({"source": str(source), "page": page})

    full_context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the information is insufficient, say 'Information not found in context.' "
        "Always cite your sources using the source tags provided in the context."
    )

    response = await groq_client.chat.completions.create(
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

    is_grounded = await evaluate_groundedness(full_context, answer)

    print(f"\nGenerated Answer:\n{answer}")
    print(f"\n[Eval Check] Grounded in Context: {is_grounded}")

    return {
        "query": query,
        "answer": answer,
        "is_grounded": is_grounded,
        "sources": sources_list
    }

async def main():
    print("=" * 60)
    print("  RAG Engine with Evaluation & Anti-Hallucination Guardrails")
    print("=" * 60)

    query_1 = "According to the text, what specific events led to the formation of the Tehrik-i-Taliban Pakistan (TTP) in 2007?"
    print(f"\n--- [Test 1: Valid Query] ---")
    print(f"Query: {query_1}")
    
    result_1 = await run_rag_pipeline(query_1)
    print(f"\nResponse:\n{result_1}")

    print("\n" + "-" * 60)

    query_2 = "What is the capital of France and its population?"
    print(f"\n--- [Test 2: Out-of-Domain Query] ---")
    print(f"Query: {query_2}")
    
    result_2 = await run_rag_pipeline(query_2)
    print(f"\nResponse:\n{result_2}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())