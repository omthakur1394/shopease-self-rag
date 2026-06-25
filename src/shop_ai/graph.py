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

    f"PRODUCT CATALOG (retrieved from knowledge base for this query):\n"
    f"{context_text if context_text.strip() else '[NO RESULTS RETURNED FROM CATALOG SEARCH]'}\n\n"

    f"--- HOW TO RESPOND ---\n"
    f"1. Look at the catalog above and check it against the user's request: same category AND within their stated budget.\n"
    f"2. If at least one item matches both category and budget: present it clearly with its name and price, "
    f"in plain text only. Then ask the user if they'd like to buy it. Do NOT call place_order_tool until they explicitly confirm.\n"
    f"3. If the catalog has items but NONE match the requested category or budget: tell the user plainly, in your own words, "
    f"that nothing currently available fits what they're looking for. Don't use a fixed script — just be honest and natural about it.\n"
    f"4. If the catalog section above says '[NO RESULTS RETURNED FROM CATALOG SEARCH]': do not claim there are no matching products. "
    f"Instead say there was an issue retrieving the catalog and ask the user to rephrase or try again.\n"
    f"5. If the user has already confirmed they want to buy a specific item mentioned earlier in this conversation, "
    f"call place_order_tool using the exact product name and price discussed.\n\n"

    f"--- FORMATTING ---\n"
    f"Never output URLs, links, or image markdown (no [text](url), no ![alt](url)). "
    f"Only output product name, key specs, and price as plain text.\n\n"

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