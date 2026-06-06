from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.graph.builder import app_graph
from langgraph.errors import GraphRecursionError
import uvicorn 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class chat_bot(BaseModel):
    thread_id : str ="1"
    chat : str

@app.post("/chat")
async def chat(request:chat_bot):
    try:
        config = {"configurable":{"thread_id":request.thread_id},"recursion_limit": 3}
        res = app_graph.invoke({"question":request.chat},config=config)
        return{
            "res":res["answer"]}
    except Exception:
        return{"res": "I am having trouble finding the exact policy details right now. Please hold while I connect you to human support."}


if __name__ == "__main__":
    uvicorn.run("src.main:app",host="0.0.0.0",port=7860,reload=True)