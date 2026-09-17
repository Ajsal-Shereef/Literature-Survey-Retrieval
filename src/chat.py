import uuid

from graph_agentic import app

def main():
    print("RAG chat (agentic). Type 'exit' or 'quit' to stop.\n")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = app.invoke({"question": question}, config=config)

        print(f"\nAssistant: {result['answer']}")
        if result.get("retries", 0) > 0:
            print(f"(reformulated question {result['retries']}x -> \"{result['question']}\")")
        print("Sources:")
        for doc in result["retrieved_docs"]:
            print(f"  - {doc['metadata']['file']} (chunk {doc['metadata']['chunk_index']})")
        print()

if __name__ == "__main__":
    main()
