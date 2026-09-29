'''Runs the full chunking strategy comparison and prints the results."'''

from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.strategy_comparison import compare_chunking_strategies, print_comparison_table

dataset = load_golden_dataset()
print(f"Comparing chunking strategies across {len(dataset)} golden test cases...")
print("(This runs the full eval 3 times — once per strategy — expect ~30-45 minutes.)\n")

comparison = compare_chunking_strategies(dataset)

print("\n\n-------------------- Strategy Comparison --------------------n")
print_comparison_table(comparison)

print("\n\nNote: single-run comparison. Small differences (a few hundredths) may")
print("reflect LLM judge/generation variance rather than a real strategy difference.")
print("Larger, consistent gaps are more likely genuine signal.")