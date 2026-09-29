'''Ingesiton Pipeline this script populated CHromaDB with data, ties it together
everyhting form Loading, Chunking, Embedding and vector store into one runnable pipelin.

FLow - Load all documents -> for each strategy chunk every doc -> embed all chunks for that strategy
-> stores them, tagged by strategy.'''


from src.ingestion.pipeline import ingest_all

if __name__ == "__main__":
    result = ingest_all()
    print(f"\n------------------- Ingestion complete -------------------")
    print(result)