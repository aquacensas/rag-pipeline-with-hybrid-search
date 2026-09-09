'''This copmbines sparse and dense retrieval resutls into one ranked list using Reciprocal Rank Fusion (RRF).
RRF Uses each chunk's RANK POSITION in rach list (not its raw score),. which sidelines the problem
that dense i.e cosine similarity, and sparse i.e BM25 often returns very different score ranges'''

from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search

# k is a damping constant- higher k flattens the impact of rank 1 vs rank 10
# standard value of k is 60

RRF_K=60

def reciprocal_rank_fusion(dense_results:list[dict],sparse_results:list[dict],dense_weight:float=0.7,sparse_weight:float=0.3)->list[dict]:
    '''Combines the two ranked list into one weighted RRF'''

    scores:dict[str,float]={}
    chunk_lookup:dict[str,float]={}

    def add_ranked_list(results:list[dict],weight:float):
        for rank,result in enumerate(results,start=1):
            '''Uses chunk_id + source-path together as a unique key,
            since chunk_id alone isnt unique across documents'''
            key=f'{result['source_path']}::{result['chunk_id']}' # To recognise same chunk in both the result
            rrf_contribution=weight*(1/(RRF_K+rank))
            scores[key]=scores.get(key,0)+rrf_contribution
            chunk_lookup[key]=result
    add_ranked_list(dense_results,dense_weight)
    add_ranked_list(sparse_results,sparse_weight)

    # sorting chunks based on list or by its combined score
    ranked_keys=sorted(scores, key=lambda k: scores[k],reverse=True)

    fused_result=[]
    for key in ranked_keys:
        result=chunk_lookup[key].copy()
        result['fused_score']=scores[key]
        fused_result.append(result)
    return fused_result

def hybrid_search(query:str, strategy:str='structural',top_k:int=10,dense_weight:float=0.7,sparse_weight:float=0.3)->list[dict]:
    ''' Perform hybrid search using either dense + sparse, or semantic re-ranking '''
    
    dense_results=dense_search(query,strategy=strategy,top_k=20)
    sparse_results=sparse_search(query,strategy=strategy,top_k=20)

    fused=reciprocal_rank_fusion(dense_results,sparse_results,dense_weight=dense_weight,sparse_weight=sparse_weight)

    #return only top_k results
    return fused[:top_k]



    
    

