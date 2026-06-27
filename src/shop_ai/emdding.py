import os
import csv
from pathlib import Path
from langchain_core.documents import Document  
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from src.shop_ai.config2 import DIR2, EMBEDDING_MODEL2, Chroma_DB2
from langchain_chroma import Chroma

def csv_fun():
    all_docs = []
    for filename in os.listdir(DIR2):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(DIR2, filename)
            def read_csv_safe(path):
                try:
                    with open(path, mode='r', encoding='utf-8') as f:
                        return list(csv.DictReader(f))
                except UnicodeDecodeError:
                    with open(path, mode='r', encoding='latin-1') as f:
                        return list(csv.DictReader(f))
            
            rows = read_csv_safe(file_path)
            
            for i, row in enumerate(rows):
                content = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
                meta = {
                    "source": file_path,
                    "row": i,
                    "product_name": str(row.get("product_name", "Unknown")),
                    "price": str(row.get("price", "0")),
                    "rating": str(row.get("rating", "0"))
                }
                
                doc = Document(page_content=content, metadata=meta)
                all_docs.append(doc)
                
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