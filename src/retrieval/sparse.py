'''Sparse Retrieval keyword search using BM25. Unlike dense retrieval this doesnt understand meaning
it rewards exact term matches and weighted on how rare the term is across the corpus. This approach is critical 
for catching the exact keywords, function name, config keys or error codes that semantic search can miss

BM25 works in memory without database, we load chunks from ChromaDB, build an indeax and cache it per strategy
this is to avoid rebuilding on very query. '''

# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from src.db.vector_store import get_collection

#Cache
bm25_cache:dict[str,dict]={}

def tokenize(text:str)->list[str]:
    #BM25 works on tokens
    return text.lower().split()

def building_index(strategy:str)->dict:
    '''This pulls every chunk for one strategy, out of ChromaDB and
    build in memory BM25 index for them  '''
    collection=get_collection()
    result=collection.get(where={'strategy':strategy})

    documents=result['documents']
    metadatas=result['metadatas']
    
    tokenized_corpus=[tokenize(doc)for doc in documents]
    bm25=BM25Okapi(tokenized_corpus)

    return {
        'bm25':bm25,
        'documents':documents,
        'metadatas':metadatas
    }

def sparse_search(query:str,strategy:str='structural',top_k:int=10)->list[dict]:
    '''Returns top k chunks with highest bm25 score for this query with 
    one chunking strategy's chunk set and avoids rebuilding it on every single call'''

    if strategy not in bm25_cache:
        bm25_cache[strategy]=building_index(strategy)

    index=bm25_cache[strategy]
    bm25=index["bm25"]

    query_tokens=tokenize(query)
    scores=bm25.get_scores(query_tokens) # one score per chunk in corpus 

    # Combine each chunk with its score and then take top_k highest chunks 
    scored = list(zip(scores,index['documents'],index['metadatas']))
    scored.sort(key=lambda x:x[0],reverse=True)
    top_results=scored[:top_k]

    results=[]
    for score,doc,meta in top_results:
        results.append(
            {
                'content':doc,
                'source_path':meta['source_path'],
                'chunk_id':meta.get('chunk_index'),
                'strategy':meta['strategy'],
                'score':float(score)
            }
        )
    return results






