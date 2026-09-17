from importlib import _bootstrap_external
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from vectorstore import search
load_dotenv()


llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key=os.environ["BASERT_KEY"], model=os.environ["BASERT_MODEL"])

class RAGState(TypedDict):
    question: str
    retrieved_docs : list
    answer : str

def retrieve(state:RAGState)->dict:
    docs = search(state["question"], k=5)
    return {"retrieved_docs": docs}

def generate(state:RAGState)-> dict:
    context = "\n\n".join(
        f"[Source: {d['metadata']['file']}, chunk {d['metadata']['chunk_index']}]\n{d['text']}"
        for d in state["retrieved_docs"]
    )

    prompt = f"""Answer the question using ONLY the context below. \
            If the context doesn't contain the answer, say "I don't know based on the provided context."

            Context:
            {context}

            Question: {state["question"]}

            Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    question = "What is policy reuse in reinforcement learning?"
    result = app.invoke({"question": question})

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for r in result["retrieved_docs"]:
        print(f"  [{r['score']:.3f}] {r['metadata']['file']} (chunk {r['metadata']['chunk_index']})")