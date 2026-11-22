---
title: ShopEase Self RAG
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---


#
- Evaluated the ShopEase self-reflecting RAG agent with RAGAS (25 queries, GPT-4o-mini judge): faithfulness 0.84 -> 0.91, answer relevancy 0.69 -> 0.83, context precision 0.79 -> 0.80, with 5.8s average end-to-end latency
# ShopEase Self-Aware RAG Agent

An autonomous, self-correcting RAG (Retrieval-Augmented Generation) customer support agent built for ShopEase. This system retrieves store policy documents, generates highly accurate answers, reflects on its own output, and rewrites queries if the initial response fails compliance checks.

## 🚀 Tech Stack
* **Core Framework:** LangGraph & LangChain
* **Backend API:** FastAPI
* **Vector Database:** ChromaDB (Local SQLite)
* **Package Manager:** `uv`
* **Deployment Environment:** Hugging Face Spaces (Docker)
* **Python Version:** 3.12

## 📂 Project Structure
```text
shopease-self-rag/
├── data/                  # Raw ShopEase policy PDFs
├── chroma_db/             # Pre-computed vector embeddings 
├── src/
│   ├── core/              # Configuration and environment setup
│   ├── graph/             # LangGraph state, nodes, and builder logic
│   ├── retriever/         # Ingestion and semantic search scripts
│   ├── cli.py             # Terminal-based testing interface
│   └── main.py            # FastAPI backend application
├── Dockerfile             # Hugging Face deployment configuration
├── .dockerignore          # Docker exclusion rules (allows chroma_db)
├── .gitignore             # Git exclusion rules (allows chroma_db)
└── requirements.txt       # Project dependencies   
