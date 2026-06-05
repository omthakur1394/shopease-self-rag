---
title: ShopEase Self RAG
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

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