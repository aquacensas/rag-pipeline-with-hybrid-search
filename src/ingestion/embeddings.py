'''Embeddings using OpenAI embedding model'''
import os 
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE=100

_client:OpenAI | None=None

def get_client()->OpenAI:
    global _client
    if _client is None:
        api_key=os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY is not set, please add it to your .env file')
        _client=OpenAI(api_key=api_key)
    return _client

def embed_texts(texts:list[str] | str, max_retries:int=3)->list[list[float]]:
    ''''''
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    client=get_client()
    all_embeddings:list[list[float]]=[]
    
    for batch_start in range(0, len(texts),BATCH_SIZE):
        batch=texts[batch_start:batch_start+BATCH_SIZE]

        for attempt in range(max_retries):
            try:
                response=client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                batch_embedding=[item.embedding for item in response.data]
                all_embeddings.extend(batch_embedding)
                break
            except Exception as e:
                if attempt==max_retries-1:
                    raise RuntimeError(f'Embedding failed after {max_retries} attempts: {e}')
                    wait=2** attempt
                    print(f'[embedding] Retry {attempt+1}/{max_retries} after error: {e}')
                    time.sleep(wait)
    return all_embeddings

def embed_text(text:str)->list[float]:
    '''Single text embedding'''
    return embed_texts([text])[0]