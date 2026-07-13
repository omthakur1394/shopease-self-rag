from typing import Annotated, List, Any
from typing_extensions import TypedDict
import operator

class ShopEaseRAGState(TypedDict):
    question: str
    search_query: str
    retrieved_docs: List[Any]
    answer: str
    chat_history: Annotated[List[str], operator.add] 
    reflection: str
    revised: bool
    attempts: int
    order_id: str