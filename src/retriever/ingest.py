from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_community.vectorstores.utils import filter_complex_metadata
from src.core.config import DIR
import os 

def load_doc():
    all_docs = []
    for filename in os.listdir(DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DIR,filename)
            loader = DoclingLoader(file_path=file_path,export_type=ExportType.MARKDOWN)
            docs = loader.load()
            all_docs.extend(docs)
    if not all_docs:
        raise ValueError("pdf not found")
    return filter_complex_metadata(all_docs)

            