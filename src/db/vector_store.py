'''This stores chunks from all the three strategies in the same collection
    tagged by strategy so later we can query them and compare them separatly'''

import chromadb
from src.ingestion.chunking import Chunk

PERSIST_DIR='data/chroma'
COLLECTION_NAME='fastapi-docs'

def get_collection():
    client=chromadb.PersistentClient(path=PERSIST_DIR)
    collection=client.get_or_create_collection(name=COLLECTION_NAME)
    
    return collection

def store_chunks(chunks:list[Chunk], embeddings:list[list[float]])-> None:
    '''Stores a batch of chunks and there precomputed embeddings into chromadb
    embeddings are computed separatly via embedding.py and passed in here'''

    if not chunks:
        return 
    
    collection=get_collection()
    
    # ChromaDB list having: id,embeddings,documents,metadata(tag for strategies)
    ids=[c.chunk_id for c in chunks]
    documents=[c.content for c in chunks]
    metadatas=[
        {
            'source_path':c.source_path,
            'strategy':c.strategy,
            'chunk_index':c.chunk_index,
            'section_heading':c.section_heading or '',
        }
        for c in chunks
    ] 

    # Avoiding dublicated 
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

def query_by_strategy(query_embedding:list[float],strategy:str,top_k:int=5)->dict:
    collection=get_collection()
    return collection.query(query_embeddings=[query_embedding],n_results=top_k,where={'strategy':strategy})

def collection_stats()->dict:
    '''How many chunks in total and how many chunks per strategy are stored'''

    collection=get_collection()
    total=collection.count()
    stats={'total':total}

    for strategy in ['fixed','structural','semantic']:
        result=collection.get(where={'strategy':strategy})
        stats[strategy]=len(result['ids'])
    return stats



    