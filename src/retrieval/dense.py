'''Dense Retrieval - semantic search over the vector store. takes a user's question,
embeds it and finds the chunks whose vectors are closest by cosine similarity.
this is searching by meaning'''

from src.ingestion.embeddings import embed_text
from src.db.vector_store import query_by_strategy

def dense_search(query:str, strategy:str='structural',top_k:int=10)->list[dict]:
    '''Embed the query and retuns the top k most semnatically similar chunks for given 
    chunking strategy'''

    query_vector=embed_text(query)
    raw_results=query_by_strategy(query_vector,strategy=strategy,top_k=top_k)

    results=[]
    documents=raw_results['documents'][0]
    metadatas=raw_results['metadatas'][0]
    distances=raw_results['distances'][0]

    for doc, meta,dist in zip(documents, metadatas, distances):
        results.append(
            {
                'content':doc,
                'source_path':meta['source_path'],
                'chunk_id':meta.get('chunk_index'),
                'strategy':meta['strategy'],
                'score':1 - dist, # Flipped the similarity (1=identical, lower= less similar) also matches the bm25 scores
            } 
        )

    return results
