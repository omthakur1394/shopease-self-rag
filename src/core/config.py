import os 
from dotenv import load_dotenv
load_dotenv ()
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
DIR = "data"
llm = "gpt-5-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Chroma_DB = "chroma_db"
if not OPENAI_API_KEY:
    raise ValueError("api is not found")


llm = ChatOpenAI(model=llm,api_key=OPENAI_API_KEY,temperature=0.1)