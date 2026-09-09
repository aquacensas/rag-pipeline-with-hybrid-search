'''LLM as a judge is used here for scoring, this takes the fused top-N candidates 
and asks the llm to score each one actual relevance to the question, then sorts based on LLM score
This catches cases where fusion reranking might not perform well or miss important chunks.
'''

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL='gpt-4o-mini'

_client:OpenAI | None=None

def get_client()->OpenAI:
    global _client
    if _client is None:
        _client=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    return _client

def score_chunk(query:str,chunk_content:str)->float:
    '''Asks LLM to score based on relevance'''
    client=get_client()
    prompt=f'''Rate how releant this document excerpt is to ansering the question on a scale of 0-10. 
    0 = completely irrelevant, 10 = directly and fully answers the question..
    
    Question:{query}
    
    Excerpt: {chunk_content[:800]}
    
    Respond strictly ONLY WITH JSON object: {{'score':<number>}}'''

    try:
        response=client.chat.completions.create(model=RERANK_MODEL,messages=[{'role':'user','content':prompt}], temperature=0,response_format={'type':'json_object'})
        result=json.loads(response.choices[0].message.content)
        return float(result.get("score", 0))
    except Exception as e:
        print(f'[reranker] Failed to score chunk:{e}')
        return 0.0

def rerank(query:str,candidates:list[dict],top_k:int=5)->list[dict]:
    '''Scores every candidate against thequery using an LLm, then return the top_k highest-scoring ones'''
    scored_candidates=[]

    for candidate in candidates:
        score=score_chunk(query,candidate['content'])
        enriched=candidate.copy()
        enriched['rerank_score']=score
        scored_candidates.append(enriched)

    scored_candidates.sort(key=lambda x:x['rerank_score'],reverse=True)

    return scored_candidates[:top_k]

