'''Generation orchestration the main entry point that ties retrieval, prompt construction and LLM
generation also implements the confidence gate: if retrieval didnt find anything sufficiently 
relevant we refuse to answer rather than risk of halucinating and we skip the costly LLm call.'''

from dataclasses import dataclass, field
from openai import OpenAI
from src.config import get_logger,OPENAI_API_KEY,GENERATION_MODEL,MIN_RETRIEVAL_CONFIDENCE,RERANK_TOP_K
from src.retrieval.fusion import hybrid_search
from src.retrieval.reranker import rerank
from src.generation.prompts import build_generation_prompt
from src.generation.citations import verify_citations
from src.generation.confidence import score_completeness, compute_composite_confidence
from src.generation.citations import CitationVerificationResult

logger=get_logger(__name__)

_client:OpenAI | None=None

@dataclass
class GenerationResult:
    query: str
    answer: str
    retrieved_chunks: list[dict]
    retrieved_confidence: float
    insufficient_context: bool
    citations_verified: bool = False
    confidence_score: float | None = None
    citation_details: CitationVerificationResult | None = None

class GenerationError(Exception):
    '''This is raised when LLm call fails (network,API error)'''
    pass

@dataclass
class GenerationResult:
    '''This generates a full answer for the question: the answer text, which chunk it 
    was grounded and whether we had enough confidence to attempt every answer'''

    query:str
    answer:str
    retrieved_chunks:list[dict]
    retrieved_confidence:float
    insufficient_context: bool
    citations_verified:bool=False
    confidence_score:float | None=None

def get_client()-> OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=OPENAI_API_KEY)
    return _client        

def compute_retrieval_confidence(chunks:list[dict])->float:
    '''A simple confidence signal'''
    if not chunks:
        return 0.0
    top_score = chunks[0].get("rerank_score")
    if top_score is None:
        top_score = chunks[0].get("fused_score")
    if top_score is None:
        top_score = chunks[0].get("score", 0.0)  # dense-only fallback
    return min(top_score / 10.0, 1.0) if top_score > 1 else top_score

def generate_answer(query:str,strategy:str='structural',top_k:int=RERANK_TOP_K,retrieval_mode:str='hybrid')-> GenerationResult:
    '''This answers a question using hybrid retrieval + reranking + grounded geenration. 
    If retrieval confidence is too low, returns a structured message i.e insufficient context, 
    instead of calling the LLM at all thus saving the cost'''

    if not query or not query.strip():
        raise ValueError('Query cannot be empty')
    logger.info(f'Generating answer for query: {query!r} (strategy={strategy})')

    if retrieval_mode == "dense_only":
        # Skip BM25, fusion, and reranking entirely — pure vector similarity
        from src.retrieval.dense import dense_search
        top_chunks = dense_search(query, strategy=strategy, top_k=top_k)
    elif retrieval_mode == "hybrid":
        fused_candidates = hybrid_search(query, strategy=strategy, top_k=20)
        top_chunks = rerank(query, fused_candidates, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")

    retrieval_confidence=compute_retrieval_confidence(top_chunks)
    logger.info(f'Retrieval confidence: {retrieval_confidence:.3f}')
    
    # refuse to answer rather than halucinating if nothing sufficiently relevant was found
    if retrieval_confidence<MIN_RETRIEVAL_CONFIDENCE:
        logger.warning(
            f'Retrieval confidence {retrieval_confidence:.3f} is below threshold '
            f'{MIN_RETRIEVAL_CONFIDENCE} - returning with insufficient-context response')
        return GenerationResult(
            query=query,
            answer=(
                "Couldn't find sufficiently relevant information in the documents provided to answer the"
                "question confidently. Please check the source document carefully or rephrase the question."
            ),
            retrieved_chunks=top_chunks,
            retrieved_confidence=retrieval_confidence,
            insufficient_context=True,   
        )

    # building a grounded prompt and calling LLm
    prompt=build_generation_prompt(query,top_chunks)
    client=get_client()

    try:
        response=client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{'role':'user','content':prompt}],
            temperature=0.2 # very low: we want consistent grounding with minimal creativity
        )
        answer_text=response.choices[0].message.content
    except Exception as e:
        logger.error(f'LLM call failed: {e}')
        raise GenerationError(f'Failed to generate answer: {str(e)}') from e
    
    logger.info(f'Successfully generated answer')

    return GenerationResult(
        query=query,
        answer=answer_text,
        retrieved_chunks=top_chunks,
        retrieved_confidence=retrieval_confidence,
        insufficient_context=False
    )

def answer_question(query:str,strategy:str='structural',top_k:int=RERANK_TOP_K,retrieval_mode: str = "hybrid")->GenerationResult:
    '''This generates a grounded answer, verify its citation, score completeness, 
    and compute one composite confidence score'''

    result=generate_answer(query,strategy=strategy,top_k=top_k,retrieval_mode=retrieval_mode)

    # If the confidence gate already refused to answer then there is no 
    # answer to verify or score then the retrieval confidence is the final answer

    if result.insufficient_context:
        result.confidence_score=result.retrieved_confidence
        return result
    
    verification = verify_citations(result.answer, result.retrieved_chunks)
    completeness = score_completeness(query, result.answer)
    result.citation_details = verification

    breakdown = compute_composite_confidence(
        retrieval_confidence=result.retrieved_confidence,
        citation_coverage=verification.coverage,
        completeness_score=completeness,
    )

    result.citations_verified = verification.coverage == 1.0
    result.confidence_score = breakdown.composite_score

    return result





