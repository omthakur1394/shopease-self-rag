import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
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
pending_collection = db.pending_escalations

TOP_K = 5
PENDING_MINUTES = 15
vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

ORDER_ID_PATTERN = re.compile(r"\b(ord-[a-z0-9]+)\b", re.IGNORECASE)
ESCALATION_PATTERN = re.compile(
    r"\besc[a-z]*|support team|human agent|real person|speak to|talk to|supervisor|manager",
    re.IGNORECASE,
)
YES_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|y|ok|okay|sure|please do|go ahead|confirm|do it|haan|ha)\b",
    re.IGNORECASE,
)
NO_PATTERN = re.compile(
    r"^\s*(no|nope|cancel|dont|don't|not now|nahi|nevermind)\b",
    re.IGNORECASE,
)
CLOSED_STATUSES = [
    "Resolved",
    "Refunded",
    "Rejected",
    "Replaced",
    "Replacement Order Placed",
    "Replacement Placed",
]


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


def is_yes(text):
    return bool(YES_PATTERN.search(text or ""))


def is_no(text):
    return bool(NO_PATTERN.search(text or ""))


def find_order_id(state, message):
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


def get_thread_id(config):
    if not config:
        return "default"
    configurable = config.get("configurable") or {}
    return str(configurable.get("thread_id") or "default")


def get_username(config):
    if not config:
        return ""
    configurable = config.get("configurable") or {}
    return str(configurable.get("username") or "").strip().lower()


async def get_pending(thread_id):
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=PENDING_MINUTES)
        return await pending_collection.find_one({
            "thread_id": thread_id,
            "created_at": {"$gte": cutoff},
        })
    except Exception as e:
        print("[pending get error]", type(e).__name__)
        return None


async def save_pending(thread_id, order_id):
    try:
        await pending_collection.delete_many({"thread_id": thread_id})
        await pending_collection.insert_one({
            "thread_id": thread_id,
            "order_id": order_id,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        print("[pending save error]", type(e).__name__)


async def clear_pending(thread_id):
    try:
        await pending_collection.delete_many({"thread_id": thread_id})
    except Exception as e:
        print("[pending clear error]", type(e).__name__)


async def create_escalation(order_data):
    try:
        order_id = order_data.get("order_id")
        existing = await tickets_collection.find_one({
            "order_id": order_id,
            "status": {"$nin": CLOSED_STATUSES},
        })
        if existing:
            return "Escalation Status: this order is already with the support team for review. The customer can follow updates in the Grievance Inbox."

        now = datetime.now(timezone.utc)
        username = order_data.get("username") or order_data.get("user_id")
        text = "Customer requested escalation of this order through the support chat."
        await tickets_collection.insert_one({
            "ticket_id": f"TKT-{uuid.uuid4().hex[:6].upper()}",
            "order_id": order_id,
            "username": username,
            "customer": username,
            "email": order_data.get("email"),
            "product_name": order_data.get("product_name"),
            "price": order_data.get("price"),
            "chat_log": text,
            "message": text,
            "status": "Pending",
            "escalation_level": 1,
            "source": "support_chat",
            "created_at": now,
            "updated_at": now,
        })
        print("[escalation] ticket created")
        return "Escalation Status: this order has been sent to the support team for review. The customer can follow updates in the Grievance Inbox."
    except Exception as e:
        print("[escalation error]", type(e).__name__)
        return ""


async def handle_support_ticket(state: ShopEaseRAGState, config: RunnableConfig = None):
    original_user_msg = state.question or ""
    thread_id = get_thread_id(config)
    username = get_username(config)
    pending = await get_pending(thread_id)

    order_id = find_order_id(state, original_user_msg)
    if pending and (is_yes(original_user_msg) or is_no(original_user_msg)):
        order_id = pending.get("order_id") or order_id

    order_data = None
    escalation_info = ""
    hub_notes_summary = ""

    if order_id:
        try:
            found = await orders_collection.find_one({"order_id": order_id})
            if found:
                owner_username = str(found.get("username") or "").strip().lower()
                owner_user_id = str(found.get("user_id") or "").strip().lower()
                if username and username in (owner_username, owner_user_id):
                    order_data = found
                    order_data.pop("_id", None)
            if order_data:
                recent_notes = await db.hub_notes.find({}).sort("created_at", -1).limit(5).to_list(length=5)
                order_region = (order_data.get("hub_region") or order_data.get("shipping_address") or "").lower()
                note_lines = []
                for n in recent_notes:
                    note_text = n.get("note")
                    note_region = (n.get("region") or "").lower()
                    if note_text and note_region and note_region in order_region:
                        note_lines.append(f"{n.get('region')} Note: {note_text}")
                if note_lines:
                    hub_notes_summary = "; ".join(note_lines)
        except Exception:
            hub_notes_summary = ""

    if not order_data:
        order_id = ""

    if order_data:
        if pending and is_yes(original_user_msg):
            escalation_info = await create_escalation(order_data)
            await clear_pending(thread_id)
        elif pending and is_no(original_user_msg):
            await clear_pending(thread_id)
            escalation_info = "Escalation Status: cancelled at the customer's request. Nothing was sent."
        elif wants_escalation(original_user_msg):
            await save_pending(thread_id, order_data.get("order_id"))
            escalation_info = "Escalation Status: waiting for customer confirmation. Nothing has been sent yet. Ask the customer to reply yes if they want this order sent to the support team."

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
        sources_label = "Policy Context and Order Context"
        order_block = f"Order Context:\n{state.order_context}\n\n"
        if state.hub_notes:
            sources_label = "Policy Context, Order Context and Hub Notes"
            order_block += f"Hub Notes:\n{state.hub_notes}\n\n"

    prompt = (
        f"You are a friendly Customer Support AI for ShopEase. Answer using ONLY the {sources_label}.\n\n"
        "RULES:\n"
        "1. Start with the direct answer to what the customer asked in the first sentence.\n"
        "2. Then explain the details that matter: the conditions, time limits, amounts, exceptions and eligibility that the context gives for this exact question.\n"
        "3. Finish with the clear next step the customer should take, such as where to click or what to upload.\n"
        "4. Write a complete, helpful answer of about 5 to 8 sentences. You may use short bullet points when there are several steps or conditions. Stay on the topic of the question and do not add unrelated policies.\n"
        "5. When an Order Context is present, mention the full Product Name, the Amount Paid and the current status.\n"
        "6. Copy numbers exactly as written in the context (days, amounts, percentages). Never guess a number.\n"
        "7. Escalation: if the Order Context has an Escalation Status line that says waiting for customer confirmation, ask the customer if they want the order sent to the support team and tell them to reply yes. Say clearly that nothing has been sent yet. If it says the order has been sent, tell the customer it has been sent to the support team and they can follow updates in the Grievance Inbox. If it says cancelled, confirm that nothing was sent. If there is no Escalation Status line, never claim anything was escalated or sent. Never mention a ticket number or reference number.\n"
        "8. If the context does not contain the answer, say what you could not find and suggest the customer use the Grievance Inbox or contact ShopEase support. Do not guess. Do not promise email, SMS or notification updates.\n"
        "9. Do not invent weather delays, logistics excuses, status events, region issues or wallet credits unless written in the context.\n"
        "10. If the order is not marked Delivered, do not ask for photos of a delivered package.\n"
        "11. Reply in the same language as the customer. Do not translate nouns like Order ID or ShopEase.\n\n"
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