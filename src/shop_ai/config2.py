import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DIR2 = str((BASE_DIR / "data2").resolve())
EMBEDDING_MODEL2 = "text-embedding-3-small"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_CSV = "shopease_products"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found in environment")

embeddings2 = OpenAIEmbeddings(
    model=EMBEDDING_MODEL2,
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)