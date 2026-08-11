from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(
        description = "The question to ask the Rag engine",
        example = "What were the economic benefits"
    )
    top_k: int = Field(
        default = 3,
        description = "How man chunks to retrieve from the chromadb collection",
        ge = 3,
        le = 10
    )

class SourceMetadata(BaseModel):
    source: str
    page: int | str

class QueryResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    sources: list[SourceMetadata]
    
class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str