"""Hospital RAG Chatbot — interactive CLI.

Usage:
    uv run python main.py
"""

from hospital_rag import get_agent


def run_chat() -> None:
    agent = get_agent()
    print("Hospital RAG Chatbot  (type 'quit' to exit)\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        answer = result["messages"][-1].content
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    run_chat()
