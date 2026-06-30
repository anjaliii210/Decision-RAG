# Agentic RAG PDF Assistant

An **Agentic Retrieval-Augmented Generation (RAG)** system that enables intelligent question answering over PDF documents using semantic search, iterative retrieval, and LLM-based reasoning.

Unlike a traditional RAG pipeline, this project introduces an orchestration layer that evaluates retrieval quality, rewrites search queries when necessary, and performs adaptive retrieval before generating the final response.

---

## Features

* PDF ingestion and automatic text chunking
* Semantic embeddings using Mistral Embeddings
* Vector storage with Qdrant
* FastAPI backend
* Event-driven workflow using Inngest
* Adaptive retrieval with query rewriting
* Retrieval quality evaluation
* Context-grounded answer generation
* Modular architecture for easy extension with additional tools

---

## Architecture

```text
                 +----------------+
                 |     User       |
                 +-------+--------+
                         |
                         v
                +------------------+
                |   Agent Planner  |
                +--------+---------+
                         |
                         v
                +------------------+
                | Vector Retrieval |
                +--------+---------+
                         |
                         v
             +-----------------------+
             | Retrieval Evaluation  |
             +----------+------------+
                        |
          +-------------+-------------+
          |                           |
          | Context Sufficient?       |
          |                           |
          +-------------+-------------+
                        |
              Yes       |       No
                        |
                        |   Rewrite Query
                        |         |
                        |         v
                        |   Vector Retrieval
                        |         |
                        +---------+
                              |
                              v
                   +--------------------+
                   |  LLM Answer Engine |
                   +---------+----------+
                             |
                             v
                         Final Answer
```

---

## Tech Stack

* Python
* FastAPI
* Inngest
* Qdrant
* Mistral AI
* LlamaIndex
* OpenAI Compatible SDK
* Docker

---

## Project Structure

```
.
├── agent.py
├── main.py
├── vector_db.py
├── data_loader.py
├── custom_types.py
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Document Ingestion

* Upload a PDF
* Extract text using LlamaIndex
* Split into semantic chunks
* Generate embeddings
* Store vectors inside Qdrant

---

### 2. Agentic Retrieval

For every user query, the agent:

1. Plans the retrieval step.
2. Performs semantic search.
3. Evaluates whether the retrieved context is sufficient.
4. Rewrites the query if retrieval quality is poor.
5. Performs retrieval again if required.
6. Generates a context-grounded answer.

---

## Workflow

```
Question
     |
     v
Planner
     |
     v
Retrieve
     |
     v
Evaluate Context
     |
+----+----+
|         |
| Good    | Poor
|         |
v         v
Answer  Rewrite Query
            |
            v
        Retrieve Again
            |
            v
          Answer
```

---

## Why Agentic RAG?

Traditional RAG follows a fixed pipeline:

```
Question
    |
Retrieve
    |
Generate
```

This project introduces adaptive retrieval:

* LLM-guided planning
* Retrieval evaluation
* Query reformulation
* Iterative search

This improves retrieval robustness while remaining lightweight and modular.

---

## Future Improvements

* Dynamic tool selection
* Metadata-aware retrieval
* Hybrid lexical + vector search
* Multi-document reasoning
* Conversation memory
* Citation generation
* Web search integration
* LangGraph workflow support

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/<your-username>/agentic-rag-pdf-assistant.git
cd agentic-rag-pdf-assistant
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

### Start Qdrant

```bash
docker compose up -d
```

### Run the server

```bash
uvicorn main:app --reload
```

---

## API Events

### PDF Ingestion

```
rag/ingest_pdf
```

### Document Query

```
rag/query_pdf_ai
```

---

## Tech Highlights

* Event-driven AI workflow using Inngest
* Semantic vector search with Qdrant
* Adaptive retrieval pipeline
* Modular agent architecture
* Context-aware LLM responses
* Production-ready backend structure

---

## License

MIT License
