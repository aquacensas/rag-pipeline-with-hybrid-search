'''Diagnostic: checks both no_answer golden cases (q004, q005) against
fixed and semantic chunking strategies to see exactly which one flipped
from "correctly refused" to "incorrectly answered", and why.'''

from src.evaluation.golden_dataset import load_golden_dataset
from src.generation.generate import generate_answer

dataset = load_golden_dataset()
no_answer_cases = [e for e in dataset if e.id in ("q004", "q005")]

for strategy in ["fixed", "structural", "semantic"]:
    print(f"\n{'=' * 70}")
    print(f"STRATEGY: {strategy}")
    print(f"{'=' * 70}")

    for entry in no_answer_cases:
        result = generate_answer(entry.question, strategy=strategy)
        status = "✓ CORRECTLY REFUSED" if result.insufficient_context else "✗ ANSWERED (should have refused)"

        print(f"\n[{entry.id}] {entry.question}")
        print(f"  Retrieval confidence: {result.retrieved_confidence:.3f}")
        print(f"  {status}")
        if not result.insufficient_context:
            print(f"  Generated answer: {result.answer[:300]}")
            print(f"  Retrieved sources: {sorted({c['source_path'] for c in result.retrieved_chunks})}")