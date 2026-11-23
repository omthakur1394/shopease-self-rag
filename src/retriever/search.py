from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from src.core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_PDF, embeddings
from src.retriever.ingest import load_doc


def getvectory():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not client.collection_exists(COLLECTION_PDF):
        client.create_collection(
            collection_name=COLLECTION_PDF,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_PDF,
        embedding=embeddings,
    )

    if client.count(COLLECTION_PDF).count == 0:
        documents = load_doc()
        vectorstore.add_documents(documents)

    return vectorstore