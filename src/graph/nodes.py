import os
import re
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from src.graph.state import ShopEaseRAGState
from src.retriever.search import getvectory
from src.core.config import llm

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

client = AsyncIOMotorClient(MONGO_URI)
db = client.shopease_db
orders_collection = db.orders

vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class ReflectionOutput(BaseModel):
    is_compliant: bool = Field(description="True if the answer is helpful and compliant, False otherwise")

async def handle_support_ticket(state: ShopEaseRAGState):
    original_user_msg = state.question or ""
    order_id = state.order_id

    if not order_id:
        match = re.search(r'\b(ord-[a-z0-9]+)\b', original_user_msg, re.IGNORECASE)
        if match:
            order_id = match.group(1).upper()

    order_data = None
    hub_notes_summary = "No active region issues."

    if order_id:
        try:
            order_data = await orders_collection.find_one({"order_id": order_id})
            if order_data:
                order_data.pop("_id", None)
                recent_notes = await db.hub_notes.find({}).sort("created_at", -1).limit(5).to_list(length=5)
                order_region = (order_data.get('hub_region') or order_data.get('shipping_address') or '').lower()
                relevant_notes = [n for n in recent_notes if n.get('note') and n.get('region', '').lower() in order_region]
                if relevant_notes:
                    hub_notes_summary = "; ".join([f"{n.get('region')} Note: {n.get('note')}" for n in relevant_notes])
        except Exception:
            hub_notes_summary = "Logistics data temporarily unavailable."

    return {
        "question": original_user_msg,
        "order_id": order_id,
        "order_context": str(order_data) if order_data else "",
        "hub_notes": hub_notes_summary,
        "chat_history": [f"User: {original_user_msg}"],
        "attempts": 0,
        "revised": False,
        "answer": "" 
    }

def retrieve_policy_docs(state: ShopEaseRAGState):
    query = state.search_query if state.search_query else state.question
    clean_query = query.replace("\x00", "").strip()
    try:
        shopease_docs = shopease_kb_retriever.invoke(clean_query)
    except Exception:
        shopease_docs = []
    return {"retrieved_docs": shopease_docs}

def generate_support_answer(state: ShopEaseRAGState):
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    history_str = "\n".join(state.chat_history)
    
    prompt = (
        "You are an expert, conversational Customer Support AI for ShopEase.\n"
        "Your goal is to solve the customer's problem step-by-step using ONLY the provided Policy Context, Order Context, and Hub Notes.\n\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "1. NO UNVERIFIED CAUSES: Do NOT invent or hallucinate external weather delays, unmentioned logistics excuses, or fake status events unless explicitly written in the provided Active Hub Notes or Order Details.\n"
        "2. NO UNAUTHORIZED COMPENSATION: Do NOT promise, invent, or offer specific monetary wallet credits or financial gifts unless explicitly authorized in the Policy Context.\n"
        "3. LOGICAL PHOTO REQUESTS: Check the order status. If the order is NOT marked as 'Delivered', do NOT ask the customer for photos of a delivered package.\n"
        "4. LANGUAGE & EMPATHY: Respond in the exact same language as the customer. Do not translate nouns like 'Order ID' or 'ShopEase'.\n"
        "5. EXPLICIT PRODUCT DETAILS: When confirming an order inquiry, ALWAYS explicitly state the full Product Name and Amount Paid from Order Details.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"Order Context:\n{state.order_context}\n\n"
        f"Hub Notes:\n{state.hub_notes}\n\n"
        f"Chat History:\n{history_str}\n\n"
        f"Latest Customer Input:\n{state.question}"
    )

    answer = llm.invoke(prompt).content.strip()
    
    return {
        "answer": answer, 
        "attempts": state.attempts + 1,
        "chat_history": [f"Assistant: {answer}"]
    }

def reflect_on_policy_compliance(state: ShopEaseRAGState):
    try:
        prompt = (
            f"Is this answer helpful and compliant with ShopEase policies?\n"
            f"Question: {state.question}\nAnswer: {state.answer}"
        )
        structured_llm = llm.with_structured_output(ReflectionOutput)
        result = structured_llm.invoke(prompt)
        return {"reflection": str(result.is_compliant), "revised": not result.is_compliant}
    except Exception:
        return {"reflection": "True", "revised": False}

def rewrite_support_query(state: ShopEaseRAGState):
    prompt = f"Optimize this query for policy retrieval: {state.question}"
    new_query = llm.invoke(prompt).content.strip()
    return {"search_query": new_query}

def finalize_response(state: ShopEaseRAGState):
    return {
        "messages": [AIMessage(content=state.answer)],
        "answer": state.answer
    }