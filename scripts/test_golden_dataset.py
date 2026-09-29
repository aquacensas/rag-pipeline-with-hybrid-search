from src.evaluation.golden_dataset import load_golden_dataset

dataset = load_golden_dataset()
for entry in dataset:
    print(f"[{entry.id}] ({entry.question_type.value}) {entry.question}")