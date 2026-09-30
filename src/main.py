import os
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
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
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set in environment variables")

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

app = FastAPI(title="ShopEase AI RAG & Sales Agent")
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
    confirm_action: bool = False
    order_id: str = ""
    user_id: str = ""
    username: str = ""

@app.get("/")
async def root():
    return {"status": "ok", "service": "shopease-api", "checkpoint_db": "mongodb"}

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
        req_username = str(request.username or current_user.username or "").strip().lower()

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
        thread_id = request.thread_id if request.thread_id != "1" else (current_user.username or "1")
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        res = await app_graph.ainvoke(
            {"question": request.chat, "order_id": request.order_id}, 
            config=config
        )
        return {"res": res.get("answer", ""), "user": current_user.username}
    except Exception as e:
        return {"res": f"I am having trouble processing your request right now. Error: {str(e)}"}

@app.post("/order")
async def order(request: order_info, current_user: Annotated[TokenData, Depends(get_Current_User)]):
    try:
        thread_id = request.thread_id if request.thread_id != "1" else (current_user.username or "1")
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        current_state = agent_app.get_state(config)
        
        if current_state.next:
            chat_text = request.chat.strip().lower()
            cancel_keywords = ["cancel", "no", "stop", "abort", "don't", "dont", "deny", "reject", "nevermind"]
            confirm_keywords = ["yes", "confirm", "approve", "proceed", "sure", "ok", "okay", "place", "buy", "go ahead", "do it", "yeah", "yep", "can"]
            
            is_cancel = any(w in chat_text for w in cancel_keywords)
            is_confirm = request.confirm_action or any(w in chat_text for w in confirm_keywords)
            
            if is_confirm and not is_cancel:
                res = await agent_app.ainvoke(None, config=config)
                return {
                    "res": res["messages"][-1].content,
                    "status": "completed",
                    "user": current_user.username
                }
            elif is_cancel:
                last_msg = current_state.values["messages"][-1]
                tool_messages = []
                if getattr(last_msg, "tool_calls", None):
                    for tc in last_msg.tool_calls:
                        tc_id = tc.get("id")
                        if tc_id:
                            tool_messages.append(ToolMessage(
                                tool_call_id=tc_id,
                                content="Order cancelled by user."
                            ))
                if tool_messages:
                    await agent_app.aupdate_state(config, {"messages": tool_messages})
                else:
                    await agent_app.aupdate_state(config, {"messages": [HumanMessage(content="Action was cancelled by the user.")]})
                return {
                    "res": "The order placement was cancelled. How else can I assist you?",
                    "status": "cancelled",
                    "user": current_user.username
                }
            else:
                last_msg = current_state.values["messages"][-1]
                tool_call = last_msg.tool_calls[0] if getattr(last_msg, "tool_calls", None) else {}
                args = tool_call.get("args", {})
                product = args.get("product_name", "the requested item")
                price = args.get("price", "")
                price_text = f" for ₹{price}" if price else ""
                return {
                    "res": f"Please confirm if you would like to place the order for **{product}**{price_text} (say 'yes' to confirm or 'cancel' to cancel).",
                    "status": "waiting_approval",
                    "requires_confirmation": True,
                    "user": current_user.username
                }

        actual_user_id = request.user_id or current_user.username or thread_id
        input_state = {
            "messages": [HumanMessage(content=request.chat)], 
            "user_id": actual_user_id
        }
        res = await agent_app.ainvoke(input_state, config=config)
        
        updated_state = agent_app.get_state(config)
        if updated_state.next:
            last_msg = updated_state.values["messages"][-1]
            tool_call = last_msg.tool_calls[0] if getattr(last_msg, "tool_calls", None) else {}
            args = tool_call.get("args", {})
            product = args.get("product_name", "the requested item")
            price = args.get("price", "")
            price_text = f" for ₹{price}" if price else ""
            
            return {
                "res": f"⚠️ Human Confirmation Required: Would you like to confirm and place your order for **{product}**{price_text}?",
                "status": "waiting_approval",
                "requires_confirmation": True,
                "user": current_user.username
            }

        return {
            "res": res["messages"][-1].content,
            "status": "completed",
            "user": current_user.username
        }
    except Exception as e:
        return {"res": f"Error: {str(e)}", "status": "error"}

@app.post("/token")
async def login(from_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    playload = {"id": 1, "username": from_data.username, "email": from_data.username}
    access_token = jwt.encode(playload, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token, "token_type": "bearer"
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=True)
