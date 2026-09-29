import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.graph import START, END, StateGraph
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

mongo_client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(mongo_client, db_name="shopease_db")

graph = StateGraph(ShopEaseRAGState)

graph.add_node("ticket_handler", handle_support_ticket)
graph.add_node("retriever", retrieve_policy_docs)
graph.add_node("responder", finalize_response)
graph.add_node("reflections", reflect_on_policy_compliance)
graph.add_node("rewrite", rewrite_support_query)
graph.add_node("generate_answer", generate_support_answer)

graph.set_entry_point("ticket_handler")

graph.add_conditional_edges(
    "ticket_handler",
    lambda s: "responder" if s.answer != "" else "retriever"
)

graph.add_edge("retriever", "generate_answer")
graph.add_edge("generate_answer", "reflections")

graph.add_conditional_edges(
    "reflections",
    lambda s: "finish" if not s.revised or s.attempts >= 2 else "loop_back",
    {
        "finish": "responder",
        "loop_back": "rewrite"
    }
)

graph.add_edge("rewrite", "retriever")
graph.add_edge("responder", END)

app_graph = graph.compile(checkpointer=checkpointer)
