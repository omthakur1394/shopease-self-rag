from src.graph.state import ShopEaseRAGState
from src.graph.builder import app_graph


if __name__ == "__main__":
    while True:
        user_input = input("Enter your query")
        inint_state = ShopEaseRAGState(question=user_input)
        config = {"configurable": {"thread_id": "session_1"}}
        result = app_graph.invoke(inint_state,config=config)
        print("\n=== Final Answer ===\n", result["answer"])
        print("\n=== Reflection Log ===\n", result["reflection"])
        print("Total Attempts:", result["attempts"])