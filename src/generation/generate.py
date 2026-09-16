'''Generation orchestration the main entry point that ties retrieval, prompt construction and LLM
generation also implements the confidence gate: if retrieval didnt find anything sufficiently 
relevant we refuse to answer rather than risk of halucinating and we skip the costly LLm call.'''

from multiprocessing import Value
from dataclasses import dataclass, field
from openai import OpenAI
from src.config import get_logger,OPENAI_API_KEY,GENERATION_MODEL,MIN_RETRIEVAL_CONFIDENCE,RERANK_TOP_K
from src.retrieval.fusion import hybrid_search
from src.retrieval.reranker import rerank
from src.generation.prompts import build_generation_prompt

logger=get_logger(__name__)

_client:OpenAI | None=None

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
    top_score=chunks[0].get('rerank_score')
    if top_score is None:
        # Fall back if reranking is not applied for some reason
        top_score=chunks[0].get('fused_score',0.0)
    # Rerank score are in the range of 0-10; normalise to 0-1 to match other confidence
    return min(top_score/10.0,1.0) if top_score >1 else top_score

def generate_answer(query:str,strategy:str='structural',top_k:int=RERANK_TOP_K) -> GenerationResult:
    '''This answers a question using hybrid retrieval + reranking + grounded geenration. 
    If retrieval confidence is too low, returns a structured message i.e insufficient context, 
    instead of calling the LLM at all thus saving the cost'''

    if not query or not query.strip():
        raise ValueError('Query cannot be empty')
    logger.info(f'Generating answer for query: {query!r} (strategy={strategy})')

    fused_candidates=hybrid_search(query,strategy=strategy,top_k=20)
    top_chunks=rerank(query,fused_candidates,top_k=top_k)

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





