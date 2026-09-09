from src.ingestion.loaders import load_corpus 

docs=load_corpus('data/raw')
print(f'Loaded {len(docs)} documents')
print('-------------------------------------------')

for doc in docs[:3]:
    print(f'{doc.source_path} ({doc.file_type}) - {len(doc.content)} chars')
    print(doc.content[:150])
    print('...........................................')