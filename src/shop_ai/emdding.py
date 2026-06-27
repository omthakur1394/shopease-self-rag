from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from src.shop_ai.config2 import DIR2, EMBEDDING_MODEL2, Chroma_DB2
from langchain_chroma import Chroma
import os 

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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL2)
    db_path = Path(Chroma_DB2)
    db_path.mkdir(parents=True, exist_ok=True)

    if any(db_path.iterdir()):
        return Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings
        )

    documents = csv_fun()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    return vectorstore

if __name__ == "__main__":
    try:
        vs = get_retiver()
        if vs:
            print("Chroma vectorstore initialized and persisted at:", Chroma_DB2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Embedding run failed:", str(e))