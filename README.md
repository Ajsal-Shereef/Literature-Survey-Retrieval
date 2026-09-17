# PhD Literature Survey RAG Assistant

This is a personal hobby project built to navigate and query my PhD candidature literature survey using an advanced Retrieval-Augmented Generation (RAG) pipeline. It is built using **LangGraph** and **FastAPI**, featuring agentic routing, self-reflection (query rewriting and document grading), and persistent memory backed by a PostgreSQL database.

## 🌟 Features
* **Agentic LangGraph Architecture**: Implements a state machine that dynamically routes between retrieving documents, grading relevance, generating answers, and rewriting queries if the initial search fails.
* **Persistent Memory**: Uses `PostgresSaver` to persist conversation state across sessions, allowing users to pause and resume multi-turn conversations reliably.
* **FastAPI Backend**: Exposes a production-ready REST API for the LangGraph agent.
* **Containerized Orchestration**: Fully Dockerized setup using `docker-compose` to network the Python API, PostgreSQL database, and local HuggingFace embedding models.

## 🚀 Quick Start (Docker)

### 1. Configure Environment Variables
This project requires API keys to run the LLM. 
First, duplicate the example environment file:
```bash
cp .env.example .env
```
Next, open the new `.env` file and insert your secret keys.

### 2. Build and Launch
Ensure you have Docker installed and running on your machine.
Run the following command from the root of the project to build the API image and start the database:

```bash
docker-compose up --build
```

### 3. Test the API
Once the containers are running and the database is healthy, you can interact with the agent by sending a POST request to the `/chat` endpoint:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "user_123", "question": "How does Align-RUDDER use reward redistribution?"}'
```

Because of the PostgreSQL checkpointer, you can run this command multiple times with the same `thread_id`, and the agent will remember your previous questions!

## 💻 Running Locally (Without Docker)
If you want to run the code locally for fast debugging:

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start a local PostgreSQL server and update your `.env` file:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
   ```
3. Run the FastAPI server:
   ```bash
   PYTHONPATH=./src uvicorn src.server:app --reload
   ```
4. Alternatively, you can run the local CLI chat interface (which uses a temporary in-memory checkpointer):
   ```bash
   PYTHONPATH=./src python -m src.chat
   ```

## 📂 Project Structure
* `src/graph_agentic.py`: The core LangGraph state machine definition and nodes.
* `src/server.py`: The FastAPI server that handles HTTP requests and manages the PostgreSQL checkpointer.
* `src/vectorstore.py`: Handles document chunking, embedding generation (via `sentence-transformers`), and similarity search (via FAISS).
* `src/chat.py`: A simple CLI tool for testing the LangGraph locally without a web server.
* `docker-compose.yml`: Orchestrates the database and API containers.
