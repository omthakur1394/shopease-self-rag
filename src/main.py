from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.graph.builder import app_graph
from langgraph.errors import GraphRecursionError
import uvicorn 
from langchain_core.messages import HumanMessage
from src.shop_ai.graph import agent_app
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
class order_info(BaseModel):
    thread_id:str = "1"
    chat:str
@app.post("/chat")
async def chat(request:chat_bot):
    try:
        config = {"configurable":{"thread_id":request.thread_id},"recursion_limit": 15}
        res = app_graph.invoke({"question":request.chat},config=config)
        return{
            "res":res["answer"]}
    except Exception:
        return{"res": "I am having trouble finding the exact policy details right now. Please hold while I connect you to human support."}
@app.post("/order")
async def order(request: order_info):
    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
        
        input_state = {
        "messages": [HumanMessage(content=request.chat)], 
        "user_id": request.thread_id}
        
        res = await agent_app.ainvoke(input_state, config=config)
        final_answer = res["messages"][-1].content
        
        return {"res": final_answer}
    except Exception as e:
        print(f"Order Error: {e}")
        return {"res": "I encountered an error while trying to process your order. Please try again."}


if __name__ == "__main__":
    uvicorn.run("src.main:app",host="0.0.0.0",port=7860,reload=True)