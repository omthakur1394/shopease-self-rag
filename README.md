# 🛍️ ShopEase Self-Aware RAG Agent

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-purple?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-green?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?style=flat)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

A production-grade, self-correcting RAG (Retrieval-Augmented Generation) customer support agent for e-commerce. Built on a stateful LangGraph workflow, the system retrieves policy documents, evaluates its own answer quality, and rewrites queries when the initial retrieval falls short — entirely without human intervention.

---

## 🧠 The Problem & Solution

Standard RAG pipelines retrieve once and generate once. If the retrieved chunks are irrelevant or incomplete, the answer is wrong — silently. For customer support, silent failures mean frustrated users and policy violations.

**ShopEase RAG solves this with an agentic self-reflection loop:**
1. Retrieves relevant policy chunks from ChromaDB.
2. Grades its own answer against the retrieved context.
3. If the answer is incomplete or hallucinated, it rewrites the query and retrieves again.
4. Only outputs the final answer after passing its own rigorous quality check.

---

## 🏗️ System Architecture

```text
User Query
│
▼
┌─────────────────────────────────────────────────┐
│              LangGraph StateGraph               │
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Retrieve │───▶│ Generate │───▶│  Reflect  │  │
│  └──────────┘    └──────────┘    └─────┬─────┘  │
│       ▲                               │         │
│       │         Rewrite Query         │         │
│       └───────────────────────────────┘         │
│                    (if quality check fails)     │
└─────────────────────────────────────────────────┘
│
▼
FastAPI Response → User