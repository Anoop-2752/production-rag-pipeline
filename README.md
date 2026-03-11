# 🧠 Production RAG Pipeline with Automated Evaluation

A production-grade Retrieval Augmented Generation (RAG) system built with LangChain, Groq, and RAGAS — featuring hybrid search, automated evaluation, and full observability via LangSmith.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

## 🎯 What Makes This Different

Most RAG projects stop at "chatbot answers questions from a PDF." This project goes further:

- **Hybrid Retrieval** — combines semantic search (FAISS) and keyword search (BM25) for better context coverage
- **Automated Evaluation** — uses RAGAS to score every answer on Faithfulness, Relevancy, Precision, and Recall
- **Production Observability** — every query is traced end-to-end via LangSmith with latency and token metrics
- **Modular Architecture** — each component (ingestion, retrieval, generation, evaluation) is independently testable

---

## 🏗️ Architecture
```
PDF Document
    ↓
Document Loader (PyMuPDF)
    ↓
Text Chunker (RecursiveCharacterTextSplitter)
    ↓
Embeddings (HuggingFace all-MiniLM-L6-v2)  ← runs locally, no API cost
    ↓
Vector Store (FAISS)
    ↓
Hybrid Retriever (Semantic 50% + BM25 50%)
    ↓
Prompt Template + Groq LLM (LLaMA 3.3 70B)
    ↓
Answer
    ↓
RAGAS Evaluation (Faithfulness · Relevancy · Precision · Recall)
    ↓
LangSmith Tracing (Latency · Tokens · Chain Breakdown)
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq — LLaMA 3.3 70B |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS |
| Keyword Search | BM25 |
| RAG Framework | LangChain |
| Evaluation | RAGAS |
| Observability | LangSmith |
| UI | Streamlit |

---

## 📊 Evaluation Metrics (RAGAS)

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are retrieved chunks actually useful? |
| **Context Recall** | Did we retrieve all necessary information? |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/production-rag-pipeline.git
cd production-rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=production-rag-pipeline
```

Get your free API keys:
- Groq: [console.groq.com](https://console.groq.com)
- LangSmith: [smith.langchain.com](https://smith.langchain.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
production-rag-pipeline/
│
├── data/                      # Place your PDF documents here
│
├── ingestion/
│   ├── document_loader.py     # PDF loading
│   └── chunker.py             # Text splitting
│
├── retrieval/
│   ├── embedder.py            # HuggingFace embeddings
│   ├── vector_store.py        # FAISS index management
│   └── retriever.py           # Hybrid search (BM25 + semantic)
│
├── generation/
│   ├── llm.py                 # Groq LLM setup
│   ├── prompt.py              # Prompt templates
│   └── chain.py               # RAG chain with LangSmith tracing
│
├── evaluation/
│   ├── metrics.py             # RAGAS metric definitions
│   └── evaluator.py           # Evaluation runner
│
├── dashboard/
│   └── ui.py                  # Streamlit dashboard
│
├── app.py                     # Entry point
├── config.py                  # Configuration and settings
├── requirements.txt           # Dependencies
└── .env                       # API keys (never committed)
```

---

## 💡 Usage

1. Open the app at `http://localhost:8501`
2. Upload any PDF using the sidebar
3. Click **Build Knowledge Base** to index it
4. Ask questions in the **Chat** tab
5. Run RAGAS evaluation in the **Evaluation** tab
6. Monitor traces at [smith.langchain.com](https://smith.langchain.com)

---

## 🔑 Key Design Decisions

**Why Hybrid Retrieval?**
Pure semantic search misses exact keyword matches. Pure BM25 misses semantic similarity. Combining both at 50/50 weight gives the best of both worlds.

**Why local embeddings?**
Using HuggingFace `all-MiniLM-L6-v2` locally means zero embedding API costs, faster indexing, and no rate limits — critical for production.

**Why RAGAS evaluation?**
Most RAG projects have no way to measure quality. RAGAS gives objective, automated scores so you can track improvements as you tune chunking, retrieval, and prompts.

---

## 📈 Sample Evaluation Results

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Sample Q1 | 0.95 | 1.00 | 0.87 | 1.00 |
| Sample Q2 | 0.89 | 1.00 | 0.95 | 1.00 |
| Sample Q3 | 0.92 | 1.00 | 0.92 | 1.00 |

---

## 🙋 Author

**Anoop Krishna** — AI/ML Engineer
- [LinkedIn](https://www.linkedin.com/in/anoopkrishna2752/)
- [Portfolio](https://anoop-portfolio-nu.vercel.app/)