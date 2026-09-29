import os
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from src.shop_ai.graph import agent_app
from src.graph.builder import app_graph 
from src.auth.auth import SECRET_KEY, ALGORITHM, get_Current_User, TokenData
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
import uvicorn 

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.shopease_db
orders_collection = db.orders

origins = [
    "https://localhost:3000",
    "http://localhost:3000",
    "https://127.0.0.1:3000",
    "http://127.0.0.1:3000",
    "https://webapp-three-drab-53.vercel.app"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class chat_bot(BaseModel):
    thread_id: str = "1"
    chat: str
    order_id: str = ""
    user_id: str = ""
    username: str = ""

class order_info(BaseModel):
    thread_id: str = "1"
    chat: str

@app.get("/")
async def root():
    return {"status": "ok", "service": "shopease-api"}

@app.post("/chat")
async def chat(request: chat_bot, current_user: Annotated[TokenData, Depends(get_Current_User)]):
    if request.order_id:
        order = await orders_collection.find_one({"order_id": request.order_id})
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Order not found. Please verify your Order ID."
            )

        order_user_id = str(order.get("user_id") or "").strip()
        order_username = str(order.get("username") or "").strip().lower()
        req_user_id = str(request.user_id or "").strip()
        req_username = str(request.username or "").strip().lower()

        is_mismatch = False
        if order_user_id and req_user_id and order_user_id != req_user_id:
            is_mismatch = True
        elif order_username and req_username and order_username != req_username:
            is_mismatch = True

        if is_mismatch:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="This order does not belong to your account."
            )

    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
        res = await app_graph.ainvoke({"question": request.chat, "order_id": request.order_id}, config=config)
        return {"res": res["answer"], "user": current_user.username}
    except Exception as e:
        return {"res": f"I am having trouble finding the exact policy details right now. Error: {str(e)}"}

@app.post("/order")
async def order(request: order_info, current_user: Annotated[TokenData, Depends(get_Current_User)]):
    try:
        config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
        input_state = {"messages": [HumanMessage(content=request.chat)], "user_id": request.thread_id}
        res = await agent_app.ainvoke(input_state, config=config)
        return {"res": res["messages"][-1].content, "user": current_user.username}
    except Exception as e:
        return {"res": f"Error: {str(e)}"}

@app.post("/token")
async def login(from_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    playload = {"id": 1, "username": from_data.username, "email": from_data.username}
    access_token = jwt.encode(playload, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token, "token_type": "bearer"
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=True)
