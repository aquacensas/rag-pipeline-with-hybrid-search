'''This combines the three indepedent signals retrieval confidence, citation coverage, answer completeness
into one score the user sees along side the answer'''

import json
from dataclasses import dataclass
from openai import OpenAI

from src.config import (
    get_logger,
    OPENAI_API_KEY,
    RERANK_MODEL,
    CONFIDENCE_RETRIEVAL_WEIGHT,
    CONFIDENCE_CITATION_WEIGHT,
    CONFIDENCE_COMPLETENESS_WEIGHT,
)

logger=get_logger(__name__)

client:OpenAI|None=None

def get_client()->OpenAI:
    global client
    if client is None:
        client=OpenAI(api_key=OPENAI_API_KEY)
    return client

@dataclass

class ConfidenceBreakdown:
    '''The three individual signals along with the final combined score'''
    retrieval_confidence:float
    citation_coverage:float
    completeness_score:float
    composite_score:float

def score_completeness(question:str,answer:str)->float:
    '''Here we ask LLM as a judge to determine does the answer address every part
    of the question or does it leave something out and returns 0.0-1.0
    
    This is deliberately a DIFFERENT question than citation verification —
    citation verification checks "is each claim true", this checks
    "is anything missing from the answer as a whole" '''

    if not question or not question.strip():
        raise ValueError('question cannot be empty')
    if not answer or not answer.strip():
        return ValueError('answer cannot be empty')
    
    prompt = f"""Rate how completely this answer addresses the question, on a scale of 0.0 to 1.0.
            0.0 = ignores the question or addresses almost none of it.
            1.0 = fully addresses every part of the question with nothing missing.

            Question: {question}

            Answer: {answer}

            Respond in JSON format with ONLY: {{"completeness": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""

    client=get_client()
    
    try:
        response=client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{'role':'user','content':prompt}],
            temperature=0,
            response_format={'type':'json_object'}
        )
        
        result=json.loads(response.choices[0].message.content)
        score = float(result.get("completeness", 0.0))
        logger.info(f"Completeness score: {score:.3f} — {result.get('reason', '')}")
        return max(0.0, min(1.0, score))  # clamp defensively to valid range
    
    except Exception as e:
        logger.error(f"Completeness scoring failed: {e}")
        return 0.0

def compute_composite_confidence(retrieval_confidence:float,citation_coverage:float,completeness_score:float)->ConfidenceBreakdown:
    '''Combines all these three scores into one confidence score.'''

    for name,value in [
        ('retrieval_confidence',retrieval_confidence),
        ('citation_coverage',citation_coverage),
        ('completeness_score',completeness_score),
    ]:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f'{name} must be between 0.0 and 1.0, got {value}')
    
    composite=(
        retrieval_confidence * CONFIDENCE_RETRIEVAL_WEIGHT
        + citation_coverage * CONFIDENCE_CITATION_WEIGHT
        + completeness_score * CONFIDENCE_COMPLETENESS_WEIGHT
    )

    logger.info(
        f"Composite confidence: {composite:.3f} "
        f"(retrieval={retrieval_confidence:.3f}, citations={citation_coverage:.3f}, "
        f"completeness={completeness_score:.3f})"
    )

    return ConfidenceBreakdown(
        retrieval_confidence=retrieval_confidence,
        citation_coverage=citation_coverage,
        completeness_score=completeness_score,
        composite_score=composite,
    )
        

        
            