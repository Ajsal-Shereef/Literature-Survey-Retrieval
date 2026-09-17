import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from graph_agentic import compile_graph

load_dotenv()

app = FastAPI()

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")

class ChatRequest(BaseModel):
    thread_id:str
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    with ConnectionPool(DB_URI, kwargs={"autocommit": True}) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        graph_app = compile_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id" : request.thread_id}}
        response = graph_app.invoke({"question" : request.question}, config)
        return {"answer" : response["answer"]}


