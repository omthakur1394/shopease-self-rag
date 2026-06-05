from src.graph.state import ShopEaseRAGState
from src.retriever.search import getvectory
from src.core.config import llm  

vectorstore = getvectory()
shopease_kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 

def retrieve_policy_docs(state: ShopEaseRAGState) -> ShopEaseRAGState:
    query = state.search_query if state.search_query else state.question
    clean_query = query.replace("\x00", "").strip()
    
    try:
        shopease_docs = shopease_kb_retriever.invoke(clean_query)
    except Exception:
        shopease_docs = []
        
    return state.model_copy(update={"retrieved_docs": shopease_docs})

def generate_support_answer(state: ShopEaseRAGState) -> ShopEaseRAGState:
    context_parts = []
    for i, doc in enumerate(state.retrieved_docs):
        source_name = doc.metadata.get('source', f'Policy Doc {i}')
        context_parts.append(f"[{i}] (Source: {source_name}): {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    prompt = (
        "You are an expert Tier 2 Customer Support AI for ShopEase.\n"
        "Answer the customer's query using ONLY the provided ShopEase Policy Context.\n"
        "Your answer MUST be structured, polite, and directly address the customer's issue.\n"
        "Do not invent policies, timelines, or refund amounts not explicitly stated in the context.\n"
        "If the context does not contain the answer, state 'I need to escalate this to a human agent as the policy is unclear.'\n"
        "Every factual claim MUST end with a source index (e.g., [0], [1]).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{state.question}"
    )
    answer = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})

def reflect_on_policy_compliance(state: ShopEaseRAGState) -> ShopEaseRAGState:
    prompt = (
        f"Review the proposed Support Answer for the Customer Question.\n"
        f"1. Does it contain bracketed citations like [0]?\n"
        f"2. Is it based ONLY on the provided ShopEase policies without hallucinating external e-commerce rules?\n"
        f"3. Is the tone appropriate for customer support?\n"
        f"Respond ONLY with 'Reflection: YES' or 'Reflection: NO' plus a brief explanation.\n\n"
        f"Question: {state.question}\nAnswer: {state.answer}"
    )
    result = llm.invoke(prompt).content
    is_ok = "reflection: yes" in result.lower()
    return state.model_copy(update={"reflection": result, "revised": not is_ok})

def rewrite_support_query(state: ShopEaseRAGState) -> ShopEaseRAGState:
    prompt = (
        f"Original Customer Query: {state.question}\n"
        f"Failure Reason: {state.reflection}\n"
        f"Write an optimized search query to retrieve the exact ShopEase policy needed. Return ONLY the query."
    )
    new_query = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"search_query": new_query})

def finalize_response(state: ShopEaseRAGState) -> ShopEaseRAGState:
    return state