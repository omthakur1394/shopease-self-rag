import os
import re
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from src.graph.state import ShopEaseRAGState
from src.retriever.search import getvectory
from src.core.config import llm

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = AsyncIOMotorClient(MONGO_URI)
db = client.shopease_db
orders_collection = db.orders

vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

async def handle_support_ticket(state: ShopEaseRAGState):
    user_msg = state.get("question", "").lower()
    order_id = state.get("order_id", "")
    
    if not order_id:
        match = re.search(r'\bord-[a-z0-9]+\b', user_msg, re.IGNORECASE)
        if match:
            order_id = match.group(0).upper()

    new_question = state.get("question")
    if order_id:
        order_data = await orders_collection.find_one({"order_id": order_id})
        if order_data:
            new_question = f"User Question: {state.get('question')}. Order Details: {order_data}"
            
    history_update = [f"User: {state.get('question')}"]
            
    return {"question": new_question, "order_id": order_id, "chat_history": history_update}

def retrieve_policy_docs(state: ShopEaseRAGState):
    query = state.get("search_query") or state.get("question")
    clean_query = query.replace("\x00", "").strip()
    try:
        shopease_docs = shopease_kb_retriever.invoke(clean_query)
    except Exception:
        shopease_docs = []
    return {"retrieved_docs": shopease_docs}

def generate_support_answer(state: ShopEaseRAGState):
    context = "\n\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
    history_str = "\n".join(state.get("chat_history", []))
    prompt = (
        "You are an expert, conversational Customer Support AI for ShopEase.\n"
        "Your goal is to solve the customer's problem step-by-step using ONLY the provided Policy Context and Order Details.\n\n"
        "RULES:\n"
        "1. Read the Policy Context and Order Details.\n"
        "2. DO NOT repeat greetings or introductory scripts if you have already said them in the Chat History.\n"
        "3. Answer the user's latest query naturally based on the flow of the conversation.\n"
        "4. DO NOT use citation brackets or technical metadata.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"Chat History:\n{history_str}\n\n"
        f"Latest Customer Input:\n{state.get('question')}"
    )
    answer = llm.invoke(prompt).content.strip()
    
    return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

def reflect_on_policy_compliance(state: ShopEaseRAGState):
    prompt = (
        f"Is this answer helpful and compliant with ShopEase policies?\n"
        f"Respond ONLY with 'Reflection: YES' or 'Reflection: NO'.\n\n"
        f"Question: {state.get('question')}\nAnswer: {state.get('answer')}"
    )
    result = llm.invoke(prompt).content
    is_ok = "reflection: yes" in result.lower()
    return {"reflection": result, "revised": not is_ok}

def rewrite_support_query(state: ShopEaseRAGState):
    prompt = f"Optimize this query for policy retrieval: {state.get('question')}"
    new_query = llm.invoke(prompt).content.strip()
    return {"search_query": new_query}

def finalize_response(state: ShopEaseRAGState):
    return {"chat_history": [f"Bot: {state.get('answer')}"]}