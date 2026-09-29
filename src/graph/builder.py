import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.mongodb import MongoDBSaver
from src.graph.nodes import (
    reflect_on_policy_compliance, 
    retrieve_policy_docs, 
    rewrite_support_query, 
    finalize_response, 
    generate_support_answer,
    handle_support_ticket 
)
from src.graph.state import ShopEaseRAGState

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

mongo_client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(mongo_client, db_name="shopease_db")

def route_ticket(state: ShopEaseRAGState) -> str:
    if state.answer != "":
        return "responder"
    return "retriever"

def route_reflection(state: ShopEaseRAGState) -> str:
    if not state.revised or state.attempts >= 2:
        return "responder"
    return "rewrite"

graph = StateGraph(ShopEaseRAGState)

graph.add_node("ticket_handler", handle_support_ticket)
graph.add_node("retriever", retrieve_policy_docs)
graph.add_node("responder", finalize_response)
graph.add_node("reflections", reflect_on_policy_compliance)
graph.add_node("rewrite", rewrite_support_query)
graph.add_node("generate_answer", generate_support_answer)

graph.set_entry_point("ticket_handler")

graph.add_conditional_edges("ticket_handler", route_ticket)
graph.add_edge("retriever", "generate_answer")
graph.add_edge("generate_answer", "reflections")
graph.add_conditional_edges("reflections", route_reflection)
graph.add_edge("rewrite", "retriever")
graph.add_edge("responder", END)

app_graph = graph.compile(checkpointer=checkpointer)
