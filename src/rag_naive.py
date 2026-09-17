import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from vectorstore import search

load_dotenv()

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key=os.environ["BASERT_KEY"],
    model=os.environ["BASERT_MODEL"],
)

def answer(question, k=5):
    results = search(question, k=k)

    context = "\n\n".join(
        f"[Source: {r['metadata']['file']}, chunk {r['metadata']['chunk_index']}]\n{r['text']}"
        for r in results
    )

    prompt = f"""Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say "I don't know based on the provided context.""

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    return response.content, results

if __name__ == "__main__":
    question = "What is policy reuse in reinforcement learning?"
    response, sources = answer(question)

    print(f"Q: {question}\n")
    print(f"A: {response}\n")
    print("Sources:")
    for r in sources:
        print(f"  [{r['score']:.3f}] {r['metadata']['file']} (chunk {r['metadata']['chunk_index']})")
