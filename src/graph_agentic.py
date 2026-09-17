import os
import uuid
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from vectorstore import search

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
llm = ChatOpenAI(base_url = LLM_BASE_URL, api_key=os.environ["BASERT_KEY"], model=os.environ["BASERT_MODEL"])

class AgenticRAGState(TypedDict):
    question:str
    retrieved_docs:list
    is_relevant:bool
    retries: int
    answer: str
    history: Annotated[list, operator.add]

def contextualise(state:AgenticRAGState)->dict:
    history = state.get("history", [])
    if not history:
        return {"question":state["question"]}

    previous_questions = [item["question"] for item in history]
    previous_answers = [item["answer"] for item in history]
    joined_question_answer = "\n\n".join([f"Question: {q}, Answer {a}" for q,a in zip(previous_questions, previous_answers)])
    prompt = f"Condense this question {state['question']} given the conversation so far {joined_question_answer}. \
             Your output should not include any other text apart from the rewritten question."
    response = llm.invoke(prompt)
    resolved = response.content.strip()
    print(f"[contextualise] raw: {state['question']!r} -> resolved: {resolved!r}")
    return {"question": resolved, "retries": 0}


def retrieve(state:AgenticRAGState)->dict:
    return {"retrieved_docs" : search(state["question"], k=5)}

def grade_documents(state:AgenticRAGState)->dict:
    context = "\n\n".join(d["text"] for d in state["retrieved_docs"])

    prompt = f"""You are grading whether retrieved context is relevant enough to answer a question.

    Context:
    {context}

    Question: {state["question"]}

    Does the context contain information that could answer this question? \
    Respond with exactly one word: "yes" or "no"."""

    response = llm.invoke(prompt)
    return {"is_relevant": response.content.strip().lower() == "yes"}

def rewrite_query(state:AgenticRAGState)->dict:
    prompt = f"""The following question failed to retrieve relevant results from a document search. \
    Rewrite it to be more specific, or use different terminology, while keeping the same intent.

    Original question: {state["question"]}
    Your ouput should not include any other text apart from the rewritten question.
    Rewritten question:"""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    print(f"[rewrite_query] {state['question']!r} -> {rewritten!r}")
    return {"question": rewritten, "retries": state.get("retries", 0)+1}

def route_after_grading(state:AgenticRAGState)->str:
    if state["is_relevant"]:
        return "generate"
    else:
        if state.get("retries",0) < 2:
            return "rewrite_query"
        else:
            return "generate"

def generate(state:AgenticRAGState)->dict:
    context = "\n\n".join(
        f"[Source: {d['metadata']['file']}, chunk {d['metadata']['chunk_index']}]\n{d['text']}"
        for d in state["retrieved_docs"]
    )
    history = state.get("history", [])
    previous_questions = [item["question"] for item in history]
    previous_answers = [item["answer"] for item in history]

    joined_question_answer = "\n\n".join(f"[Previous Question: {q}\nPrevious Answer: {a}]" for q, a in zip(previous_questions, previous_answers))

    prompt = f"""Answer the question using ONLY the material under "Context:" below. \
            If that context doesn't contain the answer, say "I don't know based on the provided context."

            Conversation so far (for continuity only — do not treat this as a source):
            {joined_question_answer}

            Context:
            {context}

            Question: {state["question"]}

            Answer:"""

    response = llm.invoke(prompt)
    
    return {"answer": response.content, "history": [{"question": state["question"], "answer": response.content }]}

def compile_graph(checkpointer=None):
    """Compiles the graph. Defaults to MemorySaver for local testing."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

graph = StateGraph(AgenticRAGState)
graph.add_node("contextualise", contextualise)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate", generate)

graph.add_edge(START, "contextualise")
graph.add_edge("contextualise", "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grading, {"generate":"generate", "rewrite_query": "rewrite_query"})
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("generate", END)

app = compile_graph()


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    question = "How does Align-RUDDER use reward redistribution to learn from few demonstrations?"
    result = app.invoke({"question": question}, config=config)
    print("retries:", result["retries"], "is_relevant:", result["is_relevant"])

    print(f"Original question: {question}")
    print(f"Final question used: {result['question']}")
    print(f"Retries used: {result.get('retries', 0)}")
    print(f"Final grading verdict (is_relevant): {result.get('is_relevant')}\n")

    print(f"A: {result['answer']}\n")

    print("Sources:")
    for r in result["retrieved_docs"]:
        print(f"  [{r['score']:.3f}] {r['metadata']['file']} (chunk {r['metadata']['chunk_index']})")

