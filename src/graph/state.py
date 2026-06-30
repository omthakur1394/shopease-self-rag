from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document


class ShopEaseRAGState(BaseModel):
    question:str
    search_query:str = ""
    retrieved_docs: List[Document] = []
    answer: str = ""
    reflection: str = ""
    revised: bool = False
    attempts: int = 0
    order_id: str = ""

    