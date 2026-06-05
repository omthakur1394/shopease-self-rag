from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.graph.nodes import (
    reflect_on_policy_compliance,retrieve_policy_docs,rewrite_support_query,finalize_response,generate_support_answer
)
from src.graph.state import ShopEaseRAGState

graph = StateGraph(ShopEaseRAGState)

graph.add_node("retriever",retrieve_policy_docs)
graph.add_node("responder",finalize_response)
graph.add_node("reflections",reflect_on_policy_compliance)
graph.add_node("rewrite",rewrite_support_query)
graph.add_node("generate_answer",generate_support_answer)

graph.set_entry_point("retriever")
graph.add_edge("retriever","generate_answer")
graph.add_edge("generate_answer","reflections")
graph.add_conditional_edges(
    "reflections",
    lambda s: "finish" if not s.revised or s.attempts >= 2 else "loop_back",
    {
        "finish": "responder",
        "loop_back": "rewrite"
    }
)

graph.add_edge("rewrite","retriever")
graph.add_edge("responder", END)


app_graph = graph.compile(checkpointer=MemorySaver())

