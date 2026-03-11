import streamlit as st
import pandas as pd
import time
from ingestion.document_loader import load_pdf
from ingestion.chunker import chunk_documents
from retrieval.vector_store import build_vector_store, load_vector_store
from retrieval.retriever import get_retriever
from generation.chain import build_rag_chain
from evaluation.evaluator import run_evaluation
import os

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="RAG Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #080b14; }
    .block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Gradient hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.3), transparent 70%);
        border-radius: 50%;
    }
    .hero-header::after {
        content: "";
        position: absolute;
        bottom: -40px; left: 30%;
        width: 160px; height: 160px;
        background: radial-gradient(circle, rgba(139,92,246,0.2), transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.45);
        font-size: 0.9rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: rgba(255,255,255,0.5);
        font-size: 0.875rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Glassmorphism cards ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(99,102,241,0.08), rgba(139,92,246,0.04));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.5);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .metric-good  { color: #34d399; }
    .metric-warn  { color: #fbbf24; }
    .metric-bad   { color: #f87171; }
    .metric-na    { color: rgba(255,255,255,0.3); }

    /* ── Pipeline status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-active {
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.3);
        color: #34d399;
    }
    .status-inactive {
        background: rgba(248,113,113,0.1);
        border: 1px solid rgba(248,113,113,0.25);
        color: #f87171;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-active  { background: #34d399; box-shadow: 0 0 6px #34d399; }
    .dot-inactive { background: #f87171; }

    /* ── Chat messages ── */
    .stChatMessage { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.02em !important;
        transition: opacity 0.2s ease, transform 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stChatInput > div > div > textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(99,102,241,0.4);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(8, 11, 20, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.3);
        margin: 1.2rem 0 0.6rem 0;
    }
    .sidebar-logo {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Section label ── */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 0.8rem;
    }

    /* ── Architecture box ── */
    .arch-box {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        line-height: 2;
    }
    .arch-highlight { color: #a78bfa; font-weight: 600; }

    /* ── Tech badge ── */
    .tech-badge {
        display: inline-block;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #a78bfa;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────
for key, default in [
    ("retriever", None), ("chain", None), ("chunks", None),
    ("eval_df", None), ("chat_history", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ RAG Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload any PDF to build the knowledge base"
    )

    if uploaded_file:
        save_path = f"data/{uploaded_file.name}"
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("Build Knowledge Base", use_container_width=True):
            with st.spinner("Loading & chunking document…"):
                docs = load_pdf(save_path)
                chunks = chunk_documents(docs)
                st.session_state.chunks = chunks
            with st.spinner("Building vector store…"):
                build_vector_store(chunks)
            with st.spinner("Initializing pipeline…"):
                retriever = get_retriever(chunks)
                chain = build_rag_chain(retriever)
                st.session_state.retriever = retriever
                st.session_state.chain = chain
            st.success(f"{len(chunks)} chunks indexed.")

    st.markdown('<div class="sidebar-section-title">Index</div>', unsafe_allow_html=True)
    if st.button("Load Existing Index", use_container_width=True):
        try:
            with st.spinner("Loading…"):
                retriever = get_retriever(st.session_state.chunks)
                chain = build_rag_chain(retriever)
                st.session_state.retriever = retriever
                st.session_state.chain = chain
            st.success("Index loaded.")
        except Exception as e:
            st.error(str(e))

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Status</div>', unsafe_allow_html=True)
    if st.session_state.chain:
        st.markdown("""
        <div class="status-badge status-active">
            <span class="status-dot dot-active"></span> Pipeline Active
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-inactive">
            <span class="status-dot dot-inactive"></span> Pipeline Inactive
        </div>""", unsafe_allow_html=True)


# ── Hero Header ───────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">RAG Intelligence Platform</div>
    <div class="hero-subtitle">End-to-end Retrieval Augmented Generation · Hybrid Search · Automated RAGAS Evaluation</div>
</div>
""", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬  Chat", "📊  Evaluation", "🏗️  Architecture"])


# ── TAB 1: Chat ───────────────────────────────────────
with tab1:
    if not st.session_state.chain:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📄</div>
            <div style="font-size:1rem; font-weight:600; color:rgba(255,255,255,0.7); margin-bottom:0.4rem;">
                No document loaded
            </div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.35);">
                Upload a PDF in the sidebar and click <strong>Build Knowledge Base</strong> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask anything about your document…")

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner(""):
                    start = time.time()
                    answer = st.session_state.chain.invoke(question)
                    elapsed = time.time() - start
                st.markdown(answer)
                st.markdown(
                    f'<div style="font-size:0.75rem; color:rgba(255,255,255,0.25); margin-top:0.5rem;">'
                    f'⏱ {elapsed:.2f}s</div>',
                    unsafe_allow_html=True
                )
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()


# ── TAB 2: Evaluation ─────────────────────────────────
with tab2:
    if not st.session_state.chain:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📊</div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.35);">
                Build the knowledge base first to run RAGAS evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">Test Questions & Ground Truths</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            q1 = st.text_input("Question 1", placeholder="e.g. How does the system detect drowsiness?")
            q2 = st.text_input("Question 2", placeholder="e.g. What model is used for object detection?")
            q3 = st.text_input("Question 3", placeholder="e.g. How are danger levels determined?")
        with col2:
            a1 = st.text_input("Expected Answer 1", placeholder="Reference answer…")
            a2 = st.text_input("Expected Answer 2", placeholder="Reference answer…")
            a3 = st.text_input("Expected Answer 3", placeholder="Reference answer…")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Evaluation", use_container_width=True):
            questions = [q for q in [q1, q2, q3] if q.strip()]
            answers   = [a for a in [a1, a2, a3] if a.strip()]

            if not questions:
                st.warning("Enter at least one question.")
            else:
                with st.spinner("Running RAGAS evaluation — this takes 2–3 minutes…"):
                    df, results = run_evaluation(
                        st.session_state.chain,
                        st.session_state.retriever,
                        questions, answers
                    )
                    st.session_state.eval_df = df

        if st.session_state.eval_df is not None:
            df = st.session_state.eval_df
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Average Scores</div>', unsafe_allow_html=True)

            metric_defs = [
                ("Faithfulness",      "faithfulness"),
                ("Answer Relevancy",  "answer_relevancy"),
                ("Context Precision", "context_precision"),
                ("Context Recall",    "context_recall"),
            ]

            cols = st.columns(4, gap="small")
            for (label, key), col in zip(metric_defs, cols):
                with col:
                    if key in df.columns:
                        val = df[key].mean()
                        if pd.isna(val):
                            display, cls = "N/A", "metric-na"
                        elif val >= 0.8:
                            display, cls = f"{val:.3f}", "metric-good"
                        elif val >= 0.6:
                            display, cls = f"{val:.3f}", "metric-warn"
                        else:
                            display, cls = f"{val:.3f}", "metric-bad"
                    else:
                        display, cls = "N/A", "metric-na"

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value {cls}">{display}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Per-Question Breakdown</div>', unsafe_allow_html=True)

            display_cols = ["user_input"] + [k for _, k in metric_defs if k in df.columns]
            st.dataframe(
                df[display_cols].rename(columns={
                    "user_input": "Question",
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy",
                    "context_precision": "Context Precision",
                    "context_recall": "Context Recall",
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "Download Results CSV",
                df.to_csv(index=False),
                "evaluation_results.csv",
                "text/csv",
                use_container_width=True
            )


# ── TAB 3: Architecture ───────────────────────────────
with tab3:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-label">Pipeline Flow</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-box">
            <span class="arch-highlight">PDF Document</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            Document Loader <span style="color:rgba(255,255,255,0.3)">(PyMuPDF)</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            Text Chunker <span style="color:rgba(255,255,255,0.3)">(RecursiveCharacterTextSplitter)</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            Embeddings <span style="color:rgba(255,255,255,0.3)">(HuggingFace all-MiniLM-L6-v2)</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            Vector Store <span style="color:rgba(255,255,255,0.3)">(FAISS)</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            <span class="arch-highlight">Hybrid Retriever</span> <span style="color:rgba(255,255,255,0.3)">(Semantic 50% + BM25 50%)</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            Prompt Template + <span class="arch-highlight">Groq LLaMA 3.3 70B</span><br>
            &nbsp;&nbsp;&nbsp;↓<br>
            <span class="arch-highlight">Answer</span> + RAGAS Evaluation
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Tech Stack</div>', unsafe_allow_html=True)
        tech = [
            "LangChain", "Groq · LLaMA 3.3 70B", "FAISS",
            "BM25 Retriever", "HuggingFace Embeddings",
            "RAGAS Evaluation", "Streamlit", "LangSmith Tracing",
            "PyMuPDF", "sentence-transformers"
        ]
        badges = "".join(f'<span class="tech-badge">{t}</span>' for t in tech)
        st.markdown(f'<div style="line-height:2.2">{badges}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)
        metrics_info = {
            "Faithfulness": "Answer is grounded in retrieved context",
            "Answer Relevancy": "Answer directly addresses the question",
            "Context Precision": "Retrieved chunks are actually useful",
            "Context Recall": "All necessary information was retrieved",
        }
        for name, desc in metrics_info.items():
            st.markdown(f"""
            <div style="margin-bottom:0.7rem;">
                <span style="color:#a78bfa; font-weight:600; font-size:0.85rem">{name}</span>
                <div style="color:rgba(255,255,255,0.4); font-size:0.78rem; margin-top:1px">{desc}</div>
            </div>""", unsafe_allow_html=True)
