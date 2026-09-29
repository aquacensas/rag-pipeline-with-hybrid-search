'''This is the ruote handler that calls the actual endpoints, the 
validation is done in schemas.py'''

from chromadb import HttpClient
from fastapi import APIRouter,HTTPException

from src.config import get_logger
from src.generation.generate import answer_question,GenerationError
from src.db.vector_store import list_documents
from src.ingestion.pipeline import ingest_all
from src.api.schemas import (
    AskRequest, AskResponse, CitationInfo, ConfidenceInfo,
    DocumentsResponse, DocumentInfo,
    IngestRequest, IngestResponse,
)

logger=get_logger(__name__)
router=APIRouter()

@router.post("/v1/ask",response_model=AskResponse)
def ask(request: AskRequest)->AskResponse:
    '''Ask quesiton and get grounded cited confidence score answer'''
    try:
        result=answer_question(request.question,strategy=request.strategy,top_k=request.top_k)
    except GenerationError as e:
        logger.error(f'Generation failed for question {request.question!r}: {e}')
        raise HTTPException(status_code=502, detail=f'Generation faiiled: {e}')
    
    citations:list[CitationInfo]=[]
    if result.citation_details is not None:
        for claim in result.citation_details.claims:
            for num,verified in claim.verified.items():
                citations.append(CitationInfo(
                    citation_number=num,
                    source_path=result.retrieved_chunks[num - 1]["source_path"]
                        if 0 < num <= len(result.retrieved_chunks) else "unknown",
                    verified=verified,
                    reason=claim.reasoning[num]
                ))
    
    confidence = ConfidenceInfo(
        retrieval_confidence=result.retrieved_confidence,
        citation_coverage=result.citation_details.coverage if result.citation_details else None,
        composite_score=result.confidence_score,
    )

    return AskResponse(
        question=result.query,
        answer=result.answer,
        insufficient_context=result.insufficient_context,
        citations=citations,
        confidence=confidence,
        retrieved_sources=sorted({c["source_path"] for c in result.retrieved_chunks}),
    )
@router.get("/v1/documents",response_model=DocumentsResponse)
def get_documents()->DocumentsResponse:
    '''Lists every document currently indexed, with chunk count per strategy'''
    docs=list_documents()
    return DocumentsResponse(
        total_documents=len(docs),
        documents=[DocumentInfo(**d) for d in docs]
    )
@router.post("v1/ingest",response_model=IngestResponse)
def ingest(request: IngestRequest)-> IngestResponse:
    '''Triggers ingestion of document from a directory into the vector store'''
    try:
        result=ingest_all(raw_dir=request.raw_dir)
    except Exception as e:
        logger.error(f'INgestion failed: {e}')
        raise HTTPException(status_code=502,detail=f'Ingestion failed{e}:')
    return IngestResponse


        

