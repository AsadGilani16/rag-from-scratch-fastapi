from fastapi import FastAPI, HTTPException, status, UploadFile, File
from schemas import QueryRequest, QueryResponse, UploadResponse
from rag_eval_engine import run_rag_pipeline
from rag_engine import ingest_from_bytes
import traceback

app = FastAPI(
    title="RAG Engine Web API",
    description="FastAPI service exposing semantic search, guardrails, and document ingestion",
    version="1.0.0"
)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf and .txt files are supported."
        )

    try:
        contents = await file.read()
        chunk_count = ingest_from_bytes(file_bytes=contents, filename=file.filename)

        return UploadResponse(
            filename=file.filename,
            chunks_indexed=chunk_count,
            message=f"Successfully ingested {chunk_count} chunks into ChromaDB."
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@app.get("/")
def health_check():
    return {"status": "online", "message": "RAG Engine API is running"}

@app.post("/ask")
async def ask_question(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace."
        )
    try:
        response = await run_rag_pipeline(request.query, request.top_k)
        return response
    except Exception as e:
        print("\n" + "="*50)
        print("ERROR IN /ask ENDPOINT:")
        traceback.print_exc()
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))