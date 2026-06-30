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

async def handle_support_ticket(state: ShopEaseRAGState) -> ShopEaseRAGState:
    user_msg = state.question.lower()
    order_id = state.order_id
    
    if not order_id:
        match = re.search(r'\b\d{5,}\b', user_msg)
        if match:
            order_id = match.group(0)

    if order_id:
        order_data = await orders_collection.find_one({"order_id": order_id})
        if order_data:
            new_question = f"User Question: {state.question}. Order Details: {order_data}"
            return state.model_copy(update={"question": new_question, "order_id": order_id})
            
    return state

def retrieve_policy_docs(state: ShopEaseRAGState) -> ShopEaseRAGState:
    query = state.search_query if state.search_query else state.question
    clean_query = query.replace("\x00", "").strip()
    try:
        shopease_docs = shopease_kb_retriever.invoke(clean_query)
    except Exception:
        shopease_docs = []
    return state.model_copy(update={"retrieved_docs": shopease_docs})

def generate_support_answer(state: ShopEaseRAGState) -> ShopEaseRAGState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = (
        "You are an expert, conversational Customer Support AI for ShopEase.\n"
        "Your goal is to solve the customer's problem step-by-step using ONLY the provided Policy Context and Order Details.\n\n"
        "RULES:\n"
        "1. Read the Policy Context and Order Details.\n"
        "2. If the problem matches a policy (like damaged items), ask the user if they want a refund or replacement.\n"
        "3. DO NOT use citation brackets or technical metadata.\n"
        "4. Be natural and empathetic.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"Customer Input:\n{state.question}"
    )
    answer = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})

def reflect_on_policy_compliance(state: ShopEaseRAGState) -> ShopEaseRAGState:
    prompt = (
        f"Is this answer helpful and compliant with ShopEase policies?\n"
        f"Respond ONLY with 'Reflection: YES' or 'Reflection: NO'.\n\n"
        f"Question: {state.question}\nAnswer: {state.answer}"
    )
    result = llm.invoke(prompt).content
    is_ok = "reflection: yes" in result.lower()
    return state.model_copy(update={"reflection": result, "revised": not is_ok})

def rewrite_support_query(state: ShopEaseRAGState) -> ShopEaseRAGState:
    prompt = f"Optimize this query for policy retrieval: {state.question}"
    new_query = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"search_query": new_query})

def finalize_response(state: ShopEaseRAGState) -> ShopEaseRAGState:
    return state