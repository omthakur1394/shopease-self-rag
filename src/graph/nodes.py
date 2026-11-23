import os
import re
import uuid
from datetime import datetime, timezone
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
tickets_collection = db.tickets

vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ORDER_ID_PATTERN = re.compile(r"\b(ord-[a-z0-9]+)\b", re.IGNORECASE)
ESCALATE_PATTERN = re.compile(
    r"\besc[a-z]*|supervisor|manager|human agent|real person|speak to|talk to|file complaint|grievance",
    re.IGNORECASE
)
YES_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|sure|confirm|please|go ahead|do it|haan|ha|ok|okay)\b",
    re.IGNORECASE
)
NO_PATTERN = re.compile(
    r"^\s*(no|nope|cancel|don't|dont|nevermind|not now|nahi)\b",
    re.IGNORECASE
)

OPEN_STATUSES = ["Pending", "Under Investigation", "On Hold", "Paused", "Security Hold"]


class ReflectionOutput(BaseModel):
    is_compliant: bool = Field(description="True if the answer is helpful and compliant, False otherwise")


def find_order_id(state: ShopEaseRAGState, message: str) -> str:
    if state.order_id:
        return state.order_id
    match = ORDER_ID_PATTERN.search(message or "")
    if match:
        return match.group(1).upper()
    for line in reversed(state.chat_history):
        match = ORDER_ID_PATTERN.search(line)
        if match:
            return match.group(1).upper()
    return ""


def was_awaiting_escalation_confirmation(chat_history: list) -> bool:
    for line in reversed(chat_history):
        if line.startswith("Assistant:"):
            return "open a formal support ticket" in line.lower() or "reply 'yes' to proceed" in line.lower()
    return False


async def handle_support_ticket(state: ShopEaseRAGState):
    user_msg = (state.question or "").strip()
    order_id = find_order_id(state, user_msg)
    
    order_data = None
    hub_notes_summary = "No active region issues."
    escalation_status = ""

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

    if order_data:
        was_confirming = was_awaiting_escalation_confirmation(state.chat_history)
        
        if was_confirming and YES_PATTERN.search(user_msg):
            existing = await tickets_collection.find_one({"order_id": order_id, "status": {"$in": OPEN_STATUSES}})
            if existing:
                escalation_status = f"ALREADY_OPEN: Ticket {existing.get('ticket_id')} is already open and under review by Level {existing.get('escalation_level', 1)} Grievance Desk."
            else:
                now = datetime.now(timezone.utc)
                ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
                customer_user = order_data.get("username") or order_data.get("user_id") or "Customer"
                
                await tickets_collection.insert_one({
                    "ticket_id": ticket_id,
                    "order_id": order_id,
                    "username": customer_user,
                    "customer": customer_user,
                    "email": order_data.get("email"),
                    "product_name": order_data.get("product_name"),
                    "price": order_data.get("price"),
                    "message": f"Customer escalated order {order_id} via Support Chat.",
                    "status": "Pending",
                    "escalation_level": 1,
                    "source": "support_chat",
                    "created_at": now,
                    "updated_at": now,
                })
                escalation_status = f"TICKET_CREATED: Ticket {ticket_id} has been opened with our Level 1 Grievance Desk. Customer can track updates in Grievance Inbox."

        elif was_confirming and NO_PATTERN.search(user_msg):
            escalation_status = "CANCELLED: Customer chose not to open a ticket. No ticket was created."

        elif ESCALATE_PATTERN.search(user_msg):
            escalation_status = "PROMPT_CONFIRMATION: Ask customer if they would like to open a formal support ticket for this order, and tell them to reply 'yes' to proceed. Mention that nothing has been created yet."

    order_context_str = str(order_data) if order_data else ""
    if escalation_status:
        order_context_str += f"\n[Escalation Action: {escalation_status}]"

    return {
        "question": user_msg,
        "order_id": order_id,
        "order_context": order_context_str,
        "hub_notes": hub_notes_summary,
        "chat_history": [f"User: {user_msg}"],
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
        "STRICT INSTRUCTIONS:\n"
        "1. NO HALLUCINATIONS: Do not invent weather delays, warehouse excuses, or unauthorized discounts unless explicitly in the context.\n"
        "2. ORDER INQUIRIES: Only mention the full Product Name and Price in the first message when initially identifying or confirming the order. For follow-up questions, do NOT repeat the price or full title; simply refer to it as 'your order' and answer the question directly.\n"
        "3. ESCALATION HANDLING:\n"
        "   - If Order Context has '[Escalation Action: PROMPT_CONFIRMATION]', ask the customer: 'Would you like me to open a formal support ticket for your order? Please reply 'yes' to proceed.' Clearly inform them that nothing has been filed yet.\n"
        "   - If Order Context has '[Escalation Action: TICKET_CREATED]', confirm that the ticket has been officially created and is now assigned to our Grievance Desk. Remind them they can track updates in their Grievance Inbox.\n"
        "   - If Order Context has '[Escalation Action: ALREADY_OPEN]', inform the customer that their ticket is already active and under review.\n"
        "   - If Order Context has '[Escalation Action: CANCELLED]', politely acknowledge that no ticket was opened.\n"
        "4. Keep your response concise, empathetic, and professional in 3-5 sentences. Respond in the same language as the customer.\n\n"
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