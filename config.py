import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration constants ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ── Retrieval ───────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))

# ── Embeddings ──────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(BASE_DIR, "faiss_index"))

# ── Test Groq ──────────────────────────────────────────
def test_groq():
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    if not GROQ_API_KEY:
        print("❌ Groq failed: GROQ_API_KEY is not set")
        return

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL
        )
        response = llm.invoke([HumanMessage(content="Say 'Groq is working!' and nothing else.")])
        print("✅ Groq:", response.content)
    except Exception as e:
        print("❌ Groq failed:", str(e))


# ── Test LangSmith ─────────────────────────────────────
def test_langsmith():
    if not LANGCHAIN_API_KEY:
        print("❌ LangSmith failed: LANGCHAIN_API_KEY is not set")
        return

    headers = {"x-api-key": LANGCHAIN_API_KEY}

    try:
        response = requests.get(
            "https://api.smith.langchain.com/api/v1/workspaces?limit=1",
            headers=headers
        )
        if response.status_code == 200:
            print("✅ LangSmith: Connected successfully")
        else:
            print(f"❌ LangSmith failed: Status {response.status_code} — {response.text}")
    except Exception as e:
        print("❌ LangSmith failed:", str(e))


if __name__ == "__main__":
    print("\n🔍 Testing API Keys...\n")
    test_groq()
    test_langsmith()
    print("\nDone!\n")