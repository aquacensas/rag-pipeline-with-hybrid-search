'''Ingestion Pipeline - logic for loading, chunking, embedding and storing documents'''

import time 
from src.config import get_logger
from src.ingestion.loaders import load_corpus
from src.ingestion.chunking import chunk_document
from src.ingestion.embeddings import embed_texts
from src.db.vector_store import store_chunks, collection_stats

logger=get_logger(__name__)

STRATEGIES=['fixed', 'structural','semantic']

def ingest_all(raw_dir: str='data/raw')->dict:
    '''Loads all the document, chunk them in 3 different strategies, embedes them
    and stores everythign in the vector database'''

    logger.info(f'Starting ingestion from {raw_dir}')
    docs=load_corpus(raw_dir)
    logger.info(f'Loaded total {len(docs)} documents.')

    chunks_by_strategy: dict[str,int]={}

    for strategy in STRATEGIES:
        start=time.time()
        all_chunks=[]
        for doc in docs:
            if strategy=='semantic':
                chunks=chunk_document(doc,startegy=strategy,embed_fn=embed_texts)
            else:
                chunks = chunk_document(doc, strategy=strategy)
            all_chunks.extend(chunks)

        contents=[c.content for c in all_chunks]
        vectors=embed_texts(contents)
        store_chunks(all_chunks,vectors)

        chunks_by_strategy[strategy]=len(all_chunks)
        logger.info(f'{strategy}: {len(all_chunks)} chunks in {time.time()-start:.1f}s')
    
    stats=collection_stats()
    return {
        "documents_loaded": len(docs),
        "chunks_by_strategy": chunks_by_strategy,
        "collection_stats": stats,
    }

    
    