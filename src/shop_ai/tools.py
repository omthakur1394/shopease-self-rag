import os
import uuid
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = AsyncIOMotorClient(MONGO_URI)
db = client.shopease_db
orders_collection = db.orders

async def db_create_order(user_id: str, product_name: str, price: float) -> dict:
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    order_doc = {
        "order_id": order_id,
        "user_id": user_id,
        "product_name": product_name,
        "price": price,
        "status": "Placed",
        "created_at": datetime.now(timezone.utc)
    }
    await orders_collection.insert_one(order_doc)
    return order_doc

async def db_process_return(order_id: str) -> dict:
    result = await orders_collection.find_one_and_update(
        {"order_id": order_id},
        {"$set": {"status": "Returned", "returned_at": datetime.now(timezone.utc)}},
        return_document=True
    )
    return result

async def db_get_order(order_id: str) -> dict:
    return await orders_collection.find_one({"order_id": order_id})

@tool
async def place_order_tool(user_id: str, product_name: str, price: float) -> str:
    """Use this tool to automatically create and execute a purchase order for the user."""
    order = await db_create_order(user_id, product_name, price)
    return f"Order executed successfully. Order ID: {order['order_id']}, Item: {order['product_name']}, Price: {order['price']}, Status: {order['status']}."

@tool
async def return_order_tool(order_id: str) -> str:
    """Use this tool to process a return request for an order immediately."""
    order = await db_process_return(order_id)
    if not order:
        return f"Failed to return. Order ID {order_id} not found."
    return f"Return processed successfully. Order ID: {order['order_id']}, Updated Status: {order['status']}."

@tool
async def check_order_details_tool(order_id: str) -> str:
    """Use this tool to look up tracking and detailed specifications for an existing order ID."""
    order = await db_get_order(order_id)
    if not order:
        return f"No order found matching ID: {order_id}."
    return f"Order Details: ID: {order['order_id']}, Product: {order['product_name']}, Cost: {order['price']}, Current Status: {order['status']}."