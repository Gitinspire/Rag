# 🤖 RAG Document Q&A — InnovLabs.AI Knowledge Base Chatbot

A **Proof of Concept** that demonstrates how RAG (Retrieval-Augmented Generation) works by building an intelligent chatbot that answers questions from InnovLabs.AI's internal documents — an AI-Powered Scientific Discovery Platform for drug discovery, biomarker research, and laboratory operations.

---

## 🧠 What is RAG? (Explained Simply)

**RAG = Retrieval-Augmented Generation**

Imagine asking ChatGPT: *"What products does InnovLabs.AI offer?"*
ChatGPT doesn't know — it was never trained on your company's data!

### You have two options:

| Approach | How it Works | Cost | Speed |
|----------|-------------|------|-------|
| **Fine-tuning** | Retrain the model on your data | 💰💰💰 Expensive | 🐢 Days/Weeks |
| **RAG** ✅ | Search your docs first, then answer | 💰 Cheap | ⚡ Minutes |

### RAG is like an open-book exam:
- **Without RAG**: The student (LLM) answers from memory → might make things up (hallucinate)
- **With RAG**: The student (LLM) first looks up the textbook → answers accurately with citations

---

## 🔄 How RAG Works — Visual Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   PHASE 1: INDEXING (One-time setup)             │
│                                                                  │
│   📄 Documents                                                   │
│      │                                                           │
│      ▼                                                           │
│   ✂️ Chunk into smaller pieces                                   │
│      │         (500 chars each, with 50 char overlap)            │
│      ▼                                                           │
│   🧮 Convert to Embeddings (text → numbers/vectors)             │
│      │         (using Google text-embedding-004)                 │
│      ▼                                                           │
│   💾 Store in Vector Database (ChromaDB)                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               PHASE 2: QUERYING (Every question)                 │
│                                                                  │
│   ❓ User Question: "What products does InnovLabs offer?"        │
│      │                                                           │
│      ▼                                                           │
│   🧮 Convert question to Embedding                              │
│      │                                                           │
│      ▼                                                           │
│   🔍 Search Vector DB for similar chunks (top 4)                │
│      │                                                           │
│      ▼                                                           │
│   📋 Retrieved Chunks:                                           │
│      • "Biological Data Analysis Toolkit..."                     │
│      • "Electronic Lab Notebook (ELN)..."                        │
│      • "Biological Data Analysis Platform..."                    │
│      │                                                           │
│      ▼                                                           │
│   🤖 Send [Question + Chunks] to Gemini LLM                     │
│      │                                                           │
│      ▼                                                           │
│   💬 Grounded Answer (based on actual documents, not memory!)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### 1. Embeddings
Converting text into numbers (vectors) so we can do math with text.

```
"drug discovery"     → [0.12, -0.45, 0.78, 0.33, ...]  (768 numbers)
"pharma research"    → [0.11, -0.44, 0.79, 0.34, ...]  ← SIMILAR! (close vectors)
"I like pizza"       → [0.89, 0.12, -0.56, 0.01, ...]  ← DIFFERENT! (far vectors)
```

### 2. Vector Database
A database optimized for similarity search (not exact match).
- Regular DB: `WHERE name = 'John'` (exact match)
- Vector DB: "Find texts similar to this question" (semantic search)

### 3. Chunking
Breaking large documents into small pieces so we retrieve only relevant parts.

### 4. Retrieval
Finding the most relevant chunks using cosine similarity between vectors.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd llm-pocs/rag-document-qa
pip install -r requirements.txt
```

### 2. Set Up API Key
```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Google API key
# Get a key from: https://aistudio.google.com/apikey
```

### 3. Run the Chatbot
```bash
python rag_app.py
```

### 4. Ask Questions!
```
You: What products does InnovLabs offer?
You: How does AI help in pharmaceutical research?
You: What is the Electronic Lab Notebook?
You: What use cases does InnovLabs support?
You: What certifications does InnovLabs have?
You: Tell me about the team composition
You: sources              ← see which docs were used
You: quit                 ← exit
```

---

## 📁 Project Structure

```
rag-document-qa/
├── knowledge_base/              # 📄 Real InnovLabs.ai documents
│   ├── company_overview.txt     # Company info, vision, mission, contact
│   ├── ai_solutions.txt         # Products: Data Analysis Toolkit, ELN, Platform
│   ├── services.txt             # 5 service categories & sub-services
│   ├── team_and_culture.txt     # Team composition & expertise areas
│   ├── faq.txt                  # Frequently asked questions
│   └── use_cases.txt            # Domain applications & 4-step workflow
├── chroma_db/                   # 💾 Vector database (auto-created on first run)
├── rag_app.py                   # 🤖 Main RAG application (heavily commented)
├── requirements.txt             # 📦 Python dependencies
├── .env                         # 🔑 Your API key (create from .env.example)
├── .env.example                 # 📋 Template for .env
└── README.md                    # 📖 This file
```

---

## 🧪 Example Questions to Try

| Category | Question |
|----------|----------|
| Company | "What is InnovLabs.AI and what is their mission?" |
| Products | "What is the Electronic Lab Notebook?" |
| Services | "What AI solutions does InnovLabs offer?" |
| Use Cases | "How is AI used in cancer research at InnovLabs?" |
| Compliance | "What security certifications does InnovLabs have?" |
| Team | "What types of engineers work at InnovLabs?" |
| Getting Started | "How can I get started with InnovLabs services?" |

"What products does InnovLabs offer?"
"How does AI help in cancer research?"
"What certifications does InnovLabs have?"
"How can I get started with InnovLabs services?"
---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | LangChain | Orchestrates the RAG pipeline |
| LLM | Google Gemini 2.0 Flash | Generates answers |
| Embeddings | Google text-embedding-004 | Converts text to vectors |
| Vector DB | ChromaDB | Stores and searches embeddings |
| Environment | python-dotenv | Manages API keys securely |
