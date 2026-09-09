'''Ingesiton Pipeline this script populated CHromaDB with data, ties it together
everyhting form Loading, Chunking, Embedding and vector store into one runnable pipelin.

FLow - Load all documents -> for each strategy chunk every doc -> embed all chunks for that strategy
-> stores them, tagged by strategy.'''


import time

from src.ingestion.loaders import load_corpus
from src.ingestion.chunking import chunk_document
from src.ingestion.embeddings import embed_texts
from src.db.vector_store import store_chunks, collection_stats

# The three strategies we are comparing, here semantic needs embed_texts passed 
# in seince it calls the API internally to measure the sentnce similarity
STRATEGIES=['fixed','structural','semantic']

def ingest_all(raw_dir:str='data/raw')->None:
    print('loading documents...')
    docs=load_corpus(raw_dir)
    print(f'Loaded {len(docs)} documents\n')

    for strategy in STRATEGIES:
        print(f'Chunking with strategy "{strategy}"...')
        start=time.time()

        all_chunks=[]
        for doc in docs:
            if strategy=='semantic':
                chunks=chunk_document(doc,strategy=strategy, embed_fn=embed_texts)
            else:
                chunks=chunk_document(doc, strategy=strategy)
            all_chunks.extend(chunks)
        
        print(f'Produced {len(all_chunks)} chunks in {time.time() - start:.1f}s')

        # Embed every chunk content in one batch call (not per document)
        print('Embedding chunks...')
        embed_start=time.time()
        contents=[c.content for c in all_chunks]
        vectors= embed_texts(contents)
        print(f'Embedded {len(vectors)} chunks in {time.time() - embed_start:.1f}s')

        # store them tagged with strategy
        print('Storing in CHromaDB...')
        store_chunks(all_chunks,vectors)
        print(f'Done with {strategy}\n and final stat {collection_stats()}')

if __name__=='__main__':
    ingest_all()
        