from src.ingestion.loaders import load_corpus
from src.ingestion.chunking import chunk_document
from src.ingestion.embeddings import embed_texts

docs=load_corpus('data/raw')

doc=next(
    (d for d in docs if d.source_path =='tutorial/first-steps.md'), docs[0]
)

print(f'Testing semantic chunking on: {doc.source_path} ({len(doc.content)} chars)\n')

chunks= chunk_document(doc, strategy='semantic',embed_fn=embed_texts,similarity_threshold=0.75,max_chunk_size=1500)

print(f'====== semantix ({len(chunks)} chunks) =======')
for c in chunks:
    print(f'[{c.chunk_id}] {c.char_count} chars')
    print(c.content[:150])
    print('...\n')
