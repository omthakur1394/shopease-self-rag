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
escalations_collection = db.escalations

TOP_K = 5
vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

ESCALATION_PATTERN = re.compile(
    r"\besc[a-z]*|support team|human agent|real person|speak to|talk to|supervisor|manager",
    re.IGNORECASE,
)


class ReflectionOutput(BaseModel):
    is_compliant: bool = Field(
        description="True if every claim is supported by the context and the answer addresses the question, False otherwise"
    )


def docs_to_text(docs):
    parts = []
    for doc in docs:
        parts.append(doc.page_content)
    return "\n\n".join(parts)


def wants_escalation(text):
    return bool(ESCALATION_PATTERN.search(text or ""))


async def create_escalation(order_data):
    try:
        order_id = order_data.get("order_id")
        existing = await escalations_collection.find_one({"order_id": order_id, "status": "open"})
        if existing:
            ticket_id = existing["ticket_id"]
        else:
            ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
            await escalations_collection.insert_one({
                "ticket_id": ticket_id,
                "order_id": order_id,
                "user_id": order_data.get("user_id"),
                "username": order_data.get("username"),
                "reason": "customer requested escalation",
                "status": "open",
                "created_at": datetime.now(timezone.utc),
            })
        return f"Escalation Ticket: {ticket_id} is open for this order and has been sent to the support team for review."
    except Exception as e:
        print("[escalation error]", type(e).__name__)
        return ""


async def handle_support_ticket(state: ShopEaseRAGState):
    original_user_msg = state.question or ""
    order_id = state.order_id

    if not order_id:
        match = re.search(r'\b(ord-[a-z0-9]+)\b', original_user_msg, re.IGNORECASE)
        if match:
            order_id = match.group(1).upper()

    order_data = None
    escalation_info = ""
    hub_notes_summary = "No active region issues."

    if order_id:
        try:
            order_data = await orders_collection.find_one({"order_id": order_id})
            if order_data:
                order_data.pop("_id", None)
                recent_notes = await db.hub_notes.find({}).sort("created_at", -1).limit(5).to_list(length=5)
                order_region = (order_data.get("hub_region") or order_data.get("shipping_address") or "").lower()
                note_lines = []
                for n in recent_notes:
                    note_text = n.get("note")
                    note_region = (n.get("region") or "").lower()
                    if note_text and note_region in order_region:
                        note_lines.append(f"{n.get('region')} Note: {note_text}")
                if note_lines:
                    hub_notes_summary = "; ".join(note_lines)
        except Exception:
            hub_notes_summary = "Logistics data temporarily unavailable."

    if order_data and wants_escalation(original_user_msg):
        escalation_info = await create_escalation(order_data)

    order_context = ""
    if order_data:
        order_context = str(order_data)
        if escalation_info:
            order_context = order_context + "\n" + escalation_info

    return {
        "question": original_user_msg,
        "order_id": order_id,
        "order_context": order_context,
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
    except Exception as e:
        print("[retrieve error]", e)
        shopease_docs = []
    return {"retrieved_docs": shopease_docs}


def generate_support_answer(state: ShopEaseRAGState):
    context = docs_to_text(state.retrieved_docs)
    history_str = "\n".join(state.chat_history)

    sources_label = "Policy Context"
    order_block = ""
    if state.order_context:
        sources_label = "Policy Context, Order Context and Hub Notes"
        order_block = (
            f"Order Context:\n{state.order_context}\n\n"
            f"Hub Notes:\n{state.hub_notes}\n\n"
        )

    prompt = (
        f"You are a friendly Customer Support AI for ShopEase. Answer using ONLY the {sources_label}.\n\n"
        "RULES:\n"
        "1. Start with the direct answer to what the customer asked, then add the relevant order details and the next step they should take.\n"
        "2. When an Order Context is present, mention the full Product Name, the Amount Paid and the current status.\n"
        "3. Be complete and helpful in about 3 to 6 sentences. Do not add policies the customer did not ask about.\n"
        "4. Copy numbers exactly as written in the context (days, amounts, percentages). Never guess a number.\n"
        "5. If the Order Context contains an Escalation Ticket line, confirm that the order has been escalated, quote the ticket number, and say the support team will review it. Never mention a ticket that is not in the Order Context.\n"
        "6. If the context does not contain the answer, say what you could not find and suggest the customer use the Grievance Inbox or contact ShopEase support. Do not guess.\n"
        "7. Do not invent weather delays, logistics excuses, status events, or wallet credits unless written in the context.\n"
        "8. If the order is not marked Delivered, do not ask for photos of a delivered package.\n"
        "9. Reply in the same language as the customer. Do not translate nouns like Order ID or ShopEase.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"{order_block}"
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
        context = docs_to_text(state.retrieved_docs)
        prompt = (
            "Check this support answer against the Policy Context and Order Context.\n"
            "Return is_compliant = True only if EVERY factual claim and number in the answer "
            "is supported by the Policy Context or Order Context, and the answer directly "
            "addresses the question. Return False if anything is invented or unsupported.\n\n"
            f"Policy Context:\n{context}\n\n"
            f"Order Context:\n{state.order_context}\n\n"
            f"Question: {state.question}\n"
            f"Answer: {state.answer}"
        )
        structured_llm = llm.with_structured_output(ReflectionOutput)
        result = structured_llm.invoke(prompt)
        return {"reflection": str(result.is_compliant), "revised": not result.is_compliant}
    except Exception as e:
        print("[reflect error]", e)
        return {"reflection": "True", "revised": False}


def rewrite_support_query(state: ShopEaseRAGState):
    prompt = (
        "Rewrite this customer question as a short keyword search query for a retail "
        "policy knowledge base. Return ONLY the query, no explanation, no quotes.\n\n"
        f"Question: {state.question}"
    )
    new_query = llm.invoke(prompt).content.strip()
    return {"search_query": new_query}


def finalize_response(state: ShopEaseRAGState):
    return {
        "messages": [AIMessage(content=state.answer)],
        "answer": state.answer
    }