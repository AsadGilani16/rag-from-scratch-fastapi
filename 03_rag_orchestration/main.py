from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse, UploadResponse
from .rag_eval_engine import run_rag_pipeline

app = FastAPI(
    title="RAG Engine Web API",
    description="FastAPI service exposing semantic search, guardrails, and document ingestion",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "online", "message": "RAG Engine API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest) -> QueryResponse:
    try:
        result = run_rag_pipeline(query=request.query, top_k=request.top_k)
        
        return result

    except Exception as e:
        # Proper error handling if something fails inside ChromaDB or Groq
        raise HTTPException(status_code=500, detail=str(e))