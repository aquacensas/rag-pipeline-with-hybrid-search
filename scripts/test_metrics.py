'''Sanity check: run eval on a small handful of golden cases first,
covering different question types, before running the full dataset.'''

from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.metrics import run_eval

dataset = load_golden_dataset()

subset = [
    next(e for e in dataset if e.id == "q001"),  # straightforward
    next(e for e in dataset if e.id == "q003"),  # multi_hop
    next(e for e in dataset if e.id == "q004"),  # no_answer
    next(e for e in dataset if e.id == "q007"),  # ambiguous
]

report = run_eval(subset)

print(f"\n ------------- Summary -------------  ")
print(f"Mean correctness: {report.mean_correctness:.3f}")
print(f"Mean faithfulness: {report.mean_faithfulness:.3f}")
print(f"Mean retrieval relevance: {report.mean_retrieval_relevance:.3f}")
print(f"\nBreakdown by type: {report.breakdown_by_type}")

print(f"\n ------------- Per-case detail -------------")
for r in report.results:
    print(f"[{r.id}] ({r.question_type.value})")
    if r.correctness is not None:
        print(f"  Correctness: {r.correctness:.3f}")
    if r.ambiguity_handling is not None:
        print(f"  Ambiguity handling: {r.ambiguity_handling:.3f}")
    if r.faithfulness is not None:
        print(f"  Faithfulness: {r.faithfulness:.3f}")
    if r.retrieval_relevance is not None:
        print(f"  Retrieval relevance: {r.retrieval_relevance:.3f}")
    if r.correctly_refused is not None:
        print(f"  Correctly refused: {r.correctly_refused}")
    print()

print(f"\nRefusal accuracy: {report.refusal_accuracy:.3f}")
print(f"Mean ambiguity handling: {report.mean_ambiguity_handling:.3f}")