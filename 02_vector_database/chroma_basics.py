import chromadb


client = chromadb.PersistentClient(path="./my_chroma_db")

collection = client.get_or_create_collection(name="ai_knowledge_base")
#collection.modify(
#    name="ai_knowledge_base",
#)

#10 sample documents

docs = [
    "FastAPI provides fast web performance for modern Python APIs.",
    "ChromaDB stores high-dimensional embeddings for fast vector search.",
    "PyTorch gives deep control over neural network architectures.",
    "SQLAlchemy connects Python objects to relational databases like PostgreSQL.",
    "Retrieval-Augmented Generation (RAG) grounds LLMs with external context.",
    "LangGraph enables building complex stateful multi-agent workflows.",
    "Transformers rely on multi-head self-attention mechanisms.",
    "Docker containers ensure consistent environment deployments.",
    "Cosine similarity measures distance between two vector direction vectors.",
    "Vector databases index embeddings using approximate nearest neighbors."
]

metas = [
    {"topic": "backend", "difficulty": "easy"},
    {"topic": "vectordb", "difficulty": "medium"},
    {"topic": "deep_learning", "difficulty": "hard"},
    {"topic": "database", "difficulty": "easy"},
    {"topic": "rag", "difficulty": "medium"},
    {"topic": "agents", "difficulty": "hard"},
    {"topic": "nlp", "difficulty": "hard"},
    {"topic": "devops", "difficulty": "easy"},
    {"topic": "math", "difficulty": "medium"},
    {"topic": "vectordb", "difficulty": "medium"},
]

id = ["1", "2", "3", "4", "5", "6", "7", "8" ,"9" ,"10"]

collection.add(
    ids = id,
    documents = docs,
    metadatas = metas
)

response = collection.query(
    query_texts= ["How do we deploy modern software reliably?"],
    n_results= 3
)

print("\n---Query Results---")
print(response)

#now changing statements ( docs )

collection.update(
    ids = ["1", "8"],
    documents= [
        "FastAPI applications are usually deployed using Uvicorn or Gunicorn behind Nginx web servers.",
        "Docker containers and Kubernetes orchestrate continuous automated deployments in modern cloud systems."
        ],
    metadatas= [
        {"topic": "devops", "difficulty": "medium"},
        {"topic": "devops", "difficulty": "hard"}
    ]
)

updated_responses = collection.query(
    query_texts=["How do we deploy modern software reliably?"],
    n_results=2
)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(updated_responses['documents'][0], 1):
    print(f"{i}. {doc}")

