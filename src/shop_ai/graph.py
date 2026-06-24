from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
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

async def rager_assistant(state: GraphState):
    messages = list(state["messages"])
    user_query = messages[-1].content
    
    # 1. Asynchronously fetch from ChromaDB
    context_docs = await shopease_kb_retriever.ainvoke(user_query)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    
    # 2. Strict 3-Step Logic Instruction
    system_instruction = (
        f"You are an expert AI shopping assistant for ShopEase.\n\n"
        f"--- STEP 1: VERIFY MATCH & BUDGET ---\n"
        f"Look at the product information provided below. You MUST verify that the items actually match the user's request. "
        f"Check the product category AND do a strict math check on the user's budget.\n"
        f"CRITICAL GUARDRAIL: If the user asks for a 'TV under 10k', but the catalog below only has phones, OR only has TVs that cost more than 10,000, DO NOT suggest them. "
        f"Instead, politely say exactly: 'I am sorry, but right now we don't have any products in that price range.'\n\n"
        f"PRODUCT CATALOG (From ChromaDB):\n{context_text}\n\n"
        f"--- STEP 2: WAIT FOR CONFIRMATION (If Match Found) ---\n"
        f"If matching items ARE found that fit the budget, present them clearly with prices. DO NOT call the place_order_tool yet. Ask the user if they want to buy one.\n\n"
        f"--- STEP 3: EXECUTE ORDER ---\n"
        f"ONLY if the user explicitly confirms they want to buy a specific item, call the `place_order_tool`. Use the exact product name and price from the catalog text.\n\n"
        f"--- STEP 4: FORMATTING RULES ---\n"
        f"NEVER include any URLs, website links, or image tags (e.g., DO NOT output [View Product](...) or ![image](...)). Extract only the product name, specs, and price to keep the chat clean.\n\n"
        f"Active User Context ID: {state.get('user_id', 'Unknown')}"
    )
    
    # 3. Invoke LLM with memory and tools
    updated_messages = [SystemMessage(content=system_instruction)] + messages
    response = await llm_with_tools.ainvoke(updated_messages)
    return {"messages": [response]}

# Build the Graph
builder = StateGraph(GraphState)
builder.add_node("assistant", rager_assistant)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with MemorySaver to maintain chat history for confirmation steps
agent_app = builder.compile(checkpointer=MemorySaver())