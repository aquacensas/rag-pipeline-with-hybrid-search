'''Automated evaluation metrics — runs the golden dataset through the full
pipeline and scores each result. Different question types are scored on
different dimensions because "correctness" doesn't mean the same thing for
every type:

- straightforward / multi_hop: does the answer match the golden answer?
- no_answer: did the system correctly refuse rather than hallucinate?
- ambiguous: does the answer reasonably handle the ambiguity (acknowledge
  multiple interpretations, or clearly answer one valid interpretation)
  rather than being misleadingly overconfident? This is NOT the same
  question as "does it match a golden answer" — an ambiguous case's golden
  answer describes the ambiguity itself, not a single correct response, so
  comparing a real generated answer against it would always score poorly
  regardless of answer quality. Scoring it as correctness would silently
  corrupt the aggregate metric.
'''

import json
from dataclasses import dataclass

from openai import OpenAI

from src.config import get_logger, OPENAI_API_KEY, RERANK_MODEL
from src.evaluation.golden_dataset import GoldenQA, QuestionType
from src.generation.generate import generate_answer
from src.generation.citations import verify_citations

logger = get_logger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


@dataclass
class EvalCaseResult:
    '''The full scoring breakdown for one golden test case. Fields are
    Optional because not every metric applies to every question type —
    None means "not applicable to this case", never "scored zero".'''
    id: str
    question: str
    question_type: QuestionType
    generated_answer: str
    correctness: float | None          # straightforward/multi_hop only
    faithfulness: float | None         # None when there were no citations to check
    citation_accuracy: float | None    # same underlying signal as faithfulness
    retrieval_relevance: float | None  # None when there's no expected source to check
    correctly_refused: bool | None     # no_answer type only
    ambiguity_handling: float | None   # ambiguous type only


@dataclass
class EvalReport:
    '''Aggregate results across the full golden dataset run.'''
    results: list[EvalCaseResult]
    mean_correctness: float            # averaged over straightforward + multi_hop only
    mean_faithfulness: float
    mean_retrieval_relevance: float
    mean_ambiguity_handling: float     # averaged over ambiguous cases only
    refusal_accuracy: float            # fraction of no_answer cases correctly refused
    breakdown_by_type: dict[str, dict[str, float]]

def answer_declines_to_answer(answer: str) -> bool:
    '''Catches cases where the confidence gate let generation proceed, but
    the model itself recognized the context was insufficient and said so —
    this is still a correct refusal, just caught one step later than the
    gate. Simple keyword check rather than another LLM call, since this
    only needs to catch an obvious, consistent pattern from our own
    system prompt's rule 4 ("explicitly say so" when context is lacking).'''
    
    if not answer:
        return False
    lowered = answer.lower()
    decline_phrases = [
        "does not contain",
        "do not contain",
        "cannot answer",
        "can't answer",
        "not enough information",
        "insufficient information",
        "no information",
    ]
    return any(phrase in lowered for phrase in decline_phrases)


def score_answer_correctness(question: str, golden_answer: str, generated_answer: str) -> float:
    '''LLM-as-judge: does the generated answer convey the same correct
    information as the golden answer? Not an exact string match — a
    correct answer can be phrased differently.

    Only meaningful for straightforward/multi_hop questions, where a
    single golden answer represents the correct response.

    Returns 0.0-1.0. Fails safe to 0.0 on any judge error.
    '''
    if not generated_answer or not generated_answer.strip():
        return 0.0

    prompt = f"""Compare the generated answer to the golden (reference) answer for this question. Rate how well the generated answer matches the correct information in the golden answer, on a scale of 0.0 to 1.0. Respond in JSON.

0.0 = wrong or contradicts the golden answer.
1.0 = fully correct and conveys the same key information (wording can differ).

Question: {question}

Golden answer: {golden_answer}

Generated answer: {generated_answer}

Respond with ONLY: {{"correctness": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("correctness", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Correctness scoring failed: {e}")
        return 0.0


def score_ambiguity_handling(question: str, generated_answer: str) -> float:
    '''LLM-as-judge, specifically for ambiguous-type questions. Asks a
    DIFFERENT question than correctness scoring: does the answer handle
    the ambiguity well — either by acknowledging multiple valid
    interpretations, or by clearly and reasonably answering one
    interpretation — rather than being confidently misleading about a
    question that doesn't have one obvious reading?

    Returns 0.0-1.0. Fails safe to 0.0 on any judge error.
    '''
    if not generated_answer or not generated_answer.strip():
        return 0.0

    prompt = f"""This question is deliberately ambiguous — it has multiple reasonable interpretations. Rate how well the generated answer handles that ambiguity, on a scale of 0.0 to 1.0. Respond in JSON.

0.0 = the answer is confidently wrong or misleading, ignoring that other valid interpretations exist.
1.0 = the answer either clearly acknowledges multiple interpretations, or reasonably and correctly addresses at least one valid interpretation without being misleadingly definitive.

Question: {question}

Generated answer: {generated_answer}

Respond with ONLY: {{"score": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Ambiguity handling scoring failed: {e}")
        return 0.0


def score_retrieval_relevance(expected_source_paths: list[str], retrieved_chunks: list[dict]) -> float | None:
    '''Checks what fraction of the golden dataset's expected source files
    actually appear among the retrieved chunks' source paths..
    '''
    if not expected_source_paths:
        return None

    retrieved_paths = {chunk.get("source_path", "") for chunk in retrieved_chunks}
    matched = sum(1 for path in expected_source_paths if path in retrieved_paths)

    return matched / len(expected_source_paths)


def evaluate_single_case(entry: GoldenQA, strategy: str = "structural") -> EvalCaseResult:
    '''Runs one golden test case through the full pipeline and scores it
    on whichever dimensions apply to its question_type.
    '''
    logger.info(f"Evaluating {entry.id} ({entry.question_type.value}): {entry.question!r}")

    gen_result = generate_answer(entry.question, strategy=strategy)
    retrieval_relevance = score_retrieval_relevance(entry.source_paths, gen_result.retrieved_chunks)

    base_kwargs = dict(
        id=entry.id,
        question=entry.question,
        question_type=entry.question_type,
        generated_answer=gen_result.answer,
        retrieval_relevance=retrieval_relevance,
    )

    # --- no_answer: "correct" means "correctly refused" ---
    if entry.question_type == QuestionType.NO_ANSWER:
        correctly_refused = gen_result.insufficient_context or answer_declines_to_answer(gen_result.answer)
        return EvalCaseResult(
            **base_kwargs,
            correctness=None,
            faithfulness=None,
            citation_accuracy=None,
            ambiguity_handling=None,
            correctly_refused=correctly_refused,
        )

    # --- ambiguous: scored on ambiguity handling, NOT correctness ---
    if entry.question_type == QuestionType.AMBIGUOUS:
        if gen_result.insufficient_context:
            # A genuinely ambiguous question still has real, answerable
            # content behind it — a refusal here is a real failure, not
            # a neutral outcome.
            logger.warning(f"{entry.id}: system refused an ambiguous-but-answerable question")
            return EvalCaseResult(
                **base_kwargs,
                correctness=None,
                faithfulness=None,
                citation_accuracy=None,
                ambiguity_handling=0.0,
                correctly_refused=False,
            )

        ambiguity_score = score_ambiguity_handling(entry.question, gen_result.answer)
        verification = verify_citations(gen_result.answer, gen_result.retrieved_chunks)
        return EvalCaseResult(
            **base_kwargs,
            correctness=None,
            faithfulness=verification.coverage,
            citation_accuracy=verification.coverage,
            ambiguity_handling=ambiguity_score,
            correctly_refused=None,
        )

    # --- straightforward / multi_hop: refusing a real question is a failure ---
    if gen_result.insufficient_context:
        logger.warning(f"{entry.id}: system refused to answer an answerable question")
        return EvalCaseResult(
            **base_kwargs,
            correctness=0.0,
            faithfulness=None,
            citation_accuracy=None,
            ambiguity_handling=None,
            correctly_refused=False,
        )

    correctness = score_answer_correctness(entry.question, entry.golden_answer, gen_result.answer)
    verification = verify_citations(gen_result.answer, gen_result.retrieved_chunks)

    return EvalCaseResult(
        **base_kwargs,
        correctness=correctness,
        faithfulness=verification.coverage,
        citation_accuracy=verification.coverage,
        ambiguity_handling=None,
        correctly_refused=None,
    )


def _mean(values: list[float]) -> float:
    '''Average of a list, treating an empty list as 0.0 rather than crashing.'''
    return sum(values) / len(values) if values else 0.0


def run_eval(dataset: list[GoldenQA], strategy: str = "structural") -> EvalReport:
    '''Runs the full golden dataset through the pipeline and produces an
    aggregate report.
    '''
    if not dataset:
        raise ValueError("Cannot run eval on an empty dataset")

    results = [evaluate_single_case(entry, strategy=strategy) for entry in dataset]

    correctness_scores = [r.correctness for r in results if r.correctness is not None]
    faithfulness_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    relevance_scores = [r.retrieval_relevance for r in results if r.retrieval_relevance is not None]
    ambiguity_scores = [r.ambiguity_handling for r in results if r.ambiguity_handling is not None]

    no_answer_results = [r for r in results if r.question_type == QuestionType.NO_ANSWER]
    refusal_accuracy = (
        sum(1 for r in no_answer_results if r.correctly_refused) / len(no_answer_results)
        if no_answer_results else 0.0
    )

    breakdown: dict[str, dict[str, float]] = {}
    for qtype in QuestionType:
        subset = [r for r in results if r.question_type == qtype]
        if not subset:
            continue

        entry: dict[str, float] = {"count": len(subset)}

        if qtype in (QuestionType.STRAIGHTFORWARD, QuestionType.MULTI_HOP):
            entry["mean_correctness"] = _mean([r.correctness for r in subset if r.correctness is not None])
        elif qtype == QuestionType.NO_ANSWER:
            entry["refusal_accuracy"] = sum(1 for r in subset if r.correctly_refused) / len(subset)
        elif qtype == QuestionType.AMBIGUOUS:
            entry["mean_ambiguity_handling"] = _mean(
                [r.ambiguity_handling for r in subset if r.ambiguity_handling is not None]
            )

        breakdown[qtype.value] = entry

    report = EvalReport(
        results=results,
        mean_correctness=_mean(correctness_scores),
        mean_faithfulness=_mean(faithfulness_scores),
        mean_retrieval_relevance=_mean(relevance_scores),
        mean_ambiguity_handling=_mean(ambiguity_scores),
        refusal_accuracy=refusal_accuracy,
        breakdown_by_type=breakdown,
    )

    logger.info(
        f"Eval complete: {len(results)} cases — "
        f"correctness={report.mean_correctness:.3f} (straightforward+multi_hop only), "
        f"faithfulness={report.mean_faithfulness:.3f}, "
        f"retrieval_relevance={report.mean_retrieval_relevance:.3f}, "
        f"ambiguity_handling={report.mean_ambiguity_handling:.3f}, "
        f"refusal_accuracy={report.refusal_accuracy:.3f}"
    )

    return report