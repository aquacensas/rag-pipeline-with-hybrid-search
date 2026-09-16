# Central configuration 

import os 
import logging
from dotenv import load_dotenv

load_dotenv()

## logging setup
logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def get_logger(name:str) -> logging.Logger:
    return logging.getLogger(name)

# Required secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")

# Retrieval tuning
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
RRF_K = int(os.getenv("RRF_K", 60))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.7))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.3))

# Confidence thresholds
MIN_RETRIEVAL_CONFIDENCE = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", 0.15))

# Storage paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fastapi_docs")

# Composite confidence scoring weights (must sum to 1.0)
CONFIDENCE_RETRIEVAL_WEIGHT = float(os.getenv("CONFIDENCE_RETRIEVAL_WEIGHT", 0.4))
CONFIDENCE_CITATION_WEIGHT = float(os.getenv("CONFIDENCE_CITATION_WEIGHT", 0.3))
CONFIDENCE_COMPLETENESS_WEIGHT = float(os.getenv("CONFIDENCE_COMPLETENESS_WEIGHT", 0.3))