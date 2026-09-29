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
    
    user_query = next((m.content for m in reversed(messages) if m.type == "human"), "")
    
    if not user_query:
        return {"messages": []}
        
    rewrite_response = await query_rewriter.ainvoke({"query": user_query})
    search_query = rewrite_response.content.strip()
    
    context_docs = await shopease_kb_retriever.ainvoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    system_instruction = (
        f"You are an expert AI shopping assistant for ShopEase.\n\n"
        f"PRODUCT CATALOG:\n"
        f"{context_text if context_text.strip() else '[NO RESULTS RETURNED]'}\n\n"
        f"--- HUMAN-IN-THE-LOOP RULES ---\n"
        f"1. When the user indicates they want to purchase an item, select and call `place_order_tool` with the exact product name and price.\n"
        f"2. The system will automatically pause and ask the human for final approval before the order is placed.\n"
        f"3. Never output markdown images or URLs. Output product name, specs, and price as plain text.\n\n"
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

agent_app = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"]
)
