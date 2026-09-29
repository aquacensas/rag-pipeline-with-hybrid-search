'''This is a pydantic model for request/ response schemas for RAG API. This
is kept separate so that if needed it can be used by FastAPI endpoints and 
other clients to know exactly what to send and what to expect. Keeping API schemas
defined in pydantic models also ensures type safety and validation'''

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The natural language question to ask")
    strategy: str = Field(default="structural", description="Which chunking strategy's index to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to ground the answer in")
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid' (dense+BM25+rerank) or 'dense_only' (vector search alone)",
    )

class CitationInfo(BaseModel):
    citation_number: int
    source_path: str
    verified: bool
    reason: str

class ConfidenceInfo(BaseModel):
    retrieval_confidence: float
    citation_coverage: float | None
    composite_score: float | None

class AskResponse(BaseModel):
    question: str
    answer: str
    insufficient_context: bool
    citations: list[CitationInfo]
    confidence: ConfidenceInfo
    retrieved_sources: list[str]

class DocumentInfo(BaseModel):
    source_path: str
    chunks_by_strategy: dict[str, int]

class DocumentsResponse(BaseModel):
    total_documents: int
    documents: list[DocumentInfo]


class IngestRequest(BaseModel):
    raw_dir: str = Field(default="data/raw", description="Directory containing source documents")


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_by_strategy: dict[str, int]
    collection_stats: dict
