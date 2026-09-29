import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DIR2 = str((BASE_DIR / "data2").resolve())
EMBEDDING_MODEL2 = "sentence-transformers/all-MiniLM-L6-v2"
Chroma_DB2 = str((BASE_DIR / ".." / "chroma_db2").resolve())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found in environment")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)