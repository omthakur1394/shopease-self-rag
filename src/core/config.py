import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DIR = "data"
llm = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_PDF = "shopease_kb"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not found")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(model=llm, api_key=OPENAI_API_KEY, temperature=0.1)