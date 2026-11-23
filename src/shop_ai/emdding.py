import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from src.shop_ai.config2 import QDRANT_URL, QDRANT_API_KEY, COLLECTION_CSV, embeddings2, DIR2
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.utils import filter_complex_metadata


def csv_fun():
    all_docs = []
    for filename in os.listdir(DIR2):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(DIR2, filename)
            try:
                loader = CSVLoader(file_path=file_path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                loader = CSVLoader(file_path=file_path, encoding="latin-1")
                docs = loader.load()
            all_docs.extend(docs)
    if not all_docs:
        raise ValueError("csv not found")
    return filter_complex_metadata(all_docs)


def get_retiver():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not client.collection_exists(COLLECTION_CSV):
        client.create_collection(
            collection_name=COLLECTION_CSV,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_CSV,
        embedding=embeddings2,
    )

    if client.count(COLLECTION_CSV).count == 0:
        documents = csv_fun()
        vectorstore.add_documents(documents)

    return vectorstore 