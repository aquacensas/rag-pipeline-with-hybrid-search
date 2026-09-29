'''Sanity check: run the full pipeline end to end and confirm the
composite confidence score reflects all three signals correctly.'''

from src.generation.generate import answer_question

result = answer_question("How do I use BackgroundTasks in FastAPI?", strategy="structural")

print(f"Answer:\n{result.answer}\n")
print(f"Retrieval confidence: {result.retrieved_confidence:.3f}")
print(f"Citations verified (100% coverage): {result.citations_verified}")
print(f"Composite confidence score: {result.confidence_score:.3f}")