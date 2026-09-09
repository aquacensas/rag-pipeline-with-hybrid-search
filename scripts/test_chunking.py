from src.ingestion.loaders import load_corpus
from src.ingestion.chunking import chunk_document

docs = load_corpus("data/raw")
doc = next(d for d in docs if d.source_path == "tutorial/first-steps.md") if any(
    d.source_path == "tutorial/first-steps.md" for d in docs
) else docs[0]

for strategy in ["fixed", "structural"]:
    chunks = chunk_document(doc, strategy=strategy)
    print(f"\n=== {strategy} ({len(chunks)} chunks) ===")
    for c in chunks[:2]:
        print(f"[{c.chunk_id}] {c.char_count} chars — heading: {c.section_heading}")
        print(c.content[:120])
        print("...")