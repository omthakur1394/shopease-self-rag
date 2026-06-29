from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.shop_ai.config2 import llm
from src.shop_ai.emdding import get_retiver
from src.shop_ai.tools import place_order_tool, return_order_tool, check_order_details_tool

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
    
    # Safely extract ONLY the last human message for the vector search
    user_query = next((m.content for m in reversed(messages) if m.type == "human"), "")
    
    if not user_query:
        return {"messages": []}
        
    rewrite_response = await query_rewriter.ainvoke({"query": user_query})
    search_query = rewrite_response.content.strip()
    
    # Pure Text Retrieval (No Metadata extraction)
    context_docs = await shopease_kb_retriever.ainvoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Your custom, highly-optimized HITL prompt
    system_instruction = (
        f"You are an expert AI shopping assistant for ShopEase.\n\n"
        f"PRODUCT CATALOG (retrieved from knowledge base for this query):\n"
        f"{context_text if context_text.strip() else '[NO RESULTS RETURNED FROM CATALOG SEARCH]'}\n\n"
        f"--- STRICT MANDATORY WORKFLOW ---\n"
        f"You must strictly follow this two-step process for all purchases to keep the human in the loop:\n\n"
        f"STEP 1: SEARCH & PROPOSE\n"
        f"If the user asks for a product (even if they say 'I want to buy [product]'), find a match in the catalog. "
        f"Present the matching item with its exact name and price. You MUST end your message by explicitly asking: "
        f"'Would you like me to place an order for this?'\n"
        f"🛑 CRITICAL: DO NOT call the `place_order_tool` during this step. You must wait for the user to reply to your question.\n\n"
        f"STEP 2: CONFIRM & EXECUTE\n"
        f"ONLY call the `place_order_tool` if in the PREVIOUS turn you proposed a specific item, AND the user's LATEST message is a direct confirmation (e.g., 'yes', 'buy it', 'confirm', 'do it'). "
        f"If they ask a new question instead of confirming, go back to Step 1.\n\n"
        f"--- FALLBACK RULES ---\n"
        f"- If items exist but don't match the user's request, plainly state that nothing fits right now.\n"
        f"- If the catalog says '[NO RESULTS RETURNED]', say there was an issue retrieving the catalog and ask the user to rephrase.\n\n"
        f"--- FORMATTING ---\n"
        f"Never output URLs, links, or image markdown. Only output product name, key specs, and price as plain text.\n\n"
        f"Active User Context ID: {state.get('user_id', 'Unknown')}"
    )
    
    updated_messages = [SystemMessage(content=system_instruction)] + messages
    response = await llm_with_tools.ainvoke(updated_messages)
    return {"messages": [response]}

builder = StateGraph(GraphState)
builder.add_node("assistant", rager_assistant)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
agent_app = builder.compile(checkpointer=MemorySaver())