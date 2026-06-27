from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from src.shop_ai.graph import agent_app
# Ensure you have your other graph imports if you need them
from src.graph.builder import app_graph 
import uvicorn 

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Models ---
class chat_bot(BaseModel):
    thread_id: str = "1"
    chat: str

class order_info(BaseModel):
    thread_id: str = "1"
    chat: str

# --- 1. General Chat Endpoint (uses app_graph) ---
@app.post("/chat")
async def chat(request: chat_bot):
    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
        res = app_graph.invoke({"question": request.chat}, config=config)
        return {"res": res["answer"]}
    except Exception as e:
        return {"res": f"I am having trouble finding the exact policy details right now. Error: {str(e)}"}

# --- 2. Order/Shopping Agent Endpoint (uses agent_app) ---
@app.post("/order")
async def order(request: order_info):
    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
        input_state = {"messages": [HumanMessage(content=request.chat)], "user_id": request.thread_id}
        res = await agent_app.ainvoke(input_state, config=config)
        return {"res": res["messages"][-1].content}
    except Exception as e:
        return {"res": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=True)