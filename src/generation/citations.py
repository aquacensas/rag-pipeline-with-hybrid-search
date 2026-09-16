'''Checks whether each citation the LLM made in its answer actually supports the
claim it's attached to.'''

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.config import get_logger, RERANK_MODEL, OPENAI_API_KEY

logger = get_logger(__name__)
client: OpenAI | None = None


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


# Matches citation markers like [1], [2], etc. Returns list of ints.
CITATION_RE = re.compile(r'\[(\d+)\]')


@dataclass
class ClaimCitation:
    '''One claim from the answer, paired with the citation numbers attached
    to it and whether each one actually checks out.'''
    claim_text: str
    citation_numbers: list[int]
    verified: dict[int, bool]      # citation number -> supported or not
    reasoning: dict[int, str]      # citation number -> why the judge decided that


@dataclass
class CitationVerificationResult:
    '''The full verification report for one answer: every claim, every
    citation, and a summary coverage percentage.'''
    claims: list[ClaimCitation]
    total_citations: int
    verified_citations: int
    coverage: float  # verified_citations / total_citations, 0.0-1.0


def split_into_claims(answer_text: str) -> list[str]:
    '''Splits the answer into sentence-level claims. Simple period-based
    split is more than sufficient here.'''
    raw_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def extract_citation_number(claim: str) -> list[int]:
    '''Extract all citation numbers from a claim. Returns a list of ints.'''
    return [int(n) for n in CITATION_RE.findall(claim)]


def verify_single_citation(claim: str, chunk_content: str) -> tuple[bool, str]:
    '''Asks an LLM judge: does this specific chunk actually support this
    claim? Returns (is_supported, reasoning).'''
    clean_claim = CITATION_RE.sub("", claim).strip()

    prompt = f"""Does the following source text support this claim? Answer strictly with JSON.

Claim: {clean_claim}

Source text: {chunk_content[:800]}

Respond with ONLY: {{"supported": true/false, "reason": "<one short sentence>"}}"""

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            response_format={'type': 'json_object'},
        )
        result = json.loads(response.choices[0].message.content)
        return bool(result.get("supported", False)), result.get("reason", "")
    except Exception as e:
        logger.error(f'Citation verification call failed: {e}')
        # Fail safe: an unverifiable citation counts as unsupported
        return False, f'Verification failed: {e}'


def verify_citations(answer_text: str, retrieved_chunks: list[dict]) -> CitationVerificationResult:
    '''Verify every citation present in the answer against the retrieved chunks.'''
    if not answer_text or not answer_text.strip():
        raise ValueError('answer_text cannot be empty')

    claim_texts = split_into_claims(answer_text)
    claim_results: list[ClaimCitation] = []
    total_citations = 0
    verified_citations = 0

    for claim in claim_texts:
        citation_numbers = extract_citation_number(claim)
        if not citation_numbers:
            continue  # this sentence made no citation, nothing to verify

        verified_map: dict[int, bool] = {}
        reasoning_map: dict[int, str] = {}

        for citation_num in citation_numbers:
            total_citations += 1
            chunk_index = citation_num - 1  # citation [1] = chunks[0]

            if chunk_index < 0 or chunk_index >= len(retrieved_chunks):
                logger.warning(f"Citation [{citation_num}] has no matching chunk")
                verified_map[citation_num] = False
                reasoning_map[citation_num] = 'Citation number does not match any provided chunk'
                continue

            chunk_content = retrieved_chunks[chunk_index]['content']
            is_supported, reason = verify_single_citation(claim, chunk_content)

            verified_map[citation_num] = is_supported
            reasoning_map[citation_num] = reason

            if is_supported:
                verified_citations += 1

        # Append ONCE per claim, after all its citations are checked —
        # not inside the citation loop, and not skipped for later claims
        claim_results.append(ClaimCitation(
            claim_text=claim,
            citation_numbers=citation_numbers,
            verified=verified_map,
            reasoning=reasoning_map,
        ))

    # These run ONCE, after ALL claims are processed — not per-claim
    coverage = verified_citations / total_citations if total_citations > 0 else 0.0

    logger.info(
        f"Citation verification: {verified_citations}/{total_citations} "
        f"citations supported ({coverage:.1%} coverage)"
    )

    return CitationVerificationResult(
        claims=claim_results,
        total_citations=total_citations,
        verified_citations=verified_citations,
        coverage=coverage,
    )