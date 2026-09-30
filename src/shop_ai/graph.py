import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.mongodb import MongoDBSaver
from src.shop_ai.config2 import llm
from src.shop_ai.emdding import get_retiver
from src.shop_ai.tools import place_order_tool, return_order_tool, check_order_details_tool

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

mongo_client = MongoClient(MONGO_URI)
checkpointer = MongoDBSaver(mongo_client, db_name="shopease_db")

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str

vectorstore = get_retiver()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
tools = [place_order_tool, return_order_tool, check_order_details_tool]
llm_with_tools = llm.bind_tools(tools)

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an internal search query optimizer. Extract the core product or search intent from the user's query. Keep it concise."),
    ("user", "{query}")
])

query_rewriter = rewrite_prompt | llm

async def rager_assistant(state: GraphState):
    messages = list(state["messages"])
    
    user_query = next((m.content for m in reversed(messages) if m.type == "human"), "")
    
    if not user_query:
        return {"messages": []}
        
    rewrite_response = await query_rewriter.ainvoke({"query": user_query})
    search_query = rewrite_response.content.strip()
    
    context_docs = await shopease_kb_retriever.ainvoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    system_instruction = (
        f"You are an expert AI shopping assistant for ShopEase.\n\n"
        f"PRODUCT CATALOG (retrieved from knowledge base for this query):\n"
        f"{context_text if context_text.strip() else '[NO RESULTS RETURNED FROM CATALOG SEARCH]'}\n\n"
        f"1. DIRECT ORDER EXECUTION:\n"
        f"   When the user asks to buy or confirms an order for a specific product (e.g., 'yes place the order', 'buy the VU TV', 'confirm', 'yes you can place the order'), YOU MUST CALL `place_order_tool`.\n"
        f"   DO NOT ask the user for delivery address, payment method, or extra confirmation details in chat.\n"
        f"   NEVER claim that you cannot execute transactions. You HAVE full capability via `place_order_tool`.\n\n"
        f"2. AMBIGUITY CHECK:\n"
        f"   If multiple different models of the same brand exist (e.g., 50-inch vs 55-inch Kodak TV) and the user did not specify which one, ask them to clarify the exact size or model before placing the order.\n\n"
        f"FALLBACK RULES:\n"
        f"- If items exist but don't match the user's request, plainly state that nothing fits right now.\n"
        f"- If the catalog says '[NO RESULTS RETURNED]', say there was an issue retrieving the catalog and ask the user to rephrase.\n\n"
        f"FORMATTING:\n"
        f"Never output URLs, links, or image markdown. Only output product name, key specs, and price as plain text.\n\n"
        f"Active User Context ID: {state.get('user_id', 'Unknown')}"
    )
    
    sanitized_messages = []
    for i, msg in enumerate(messages):
        sanitized_messages.append(msg)
        if getattr(msg, "tool_calls", None):
            existing_tool_ids = {
                getattr(m, "tool_call_id", None) for m in messages[i+1:] if isinstance(m, ToolMessage)
            }
            for tc in msg.tool_calls:
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if tc_id and tc_id not in existing_tool_ids:
                    sanitized_messages.append(ToolMessage(
                        tool_call_id=tc_id,
                        content="Action cancelled by user."
                    ))

    updated_messages = [SystemMessage(content=system_instruction)] + sanitized_messages
    response = await llm_with_tools.ainvoke(updated_messages)
    return {"messages": [response]}

builder = StateGraph(GraphState)
builder.add_node("assistant", rager_assistant)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
agent_app = builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])
