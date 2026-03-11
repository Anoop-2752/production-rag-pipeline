import os
from dotenv import load_dotenv

load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# llama-3.3-70b gives the best quality/speed tradeoff on Groq's free tier
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 500 chars keeps chunks within the LLM's attention sweet spot
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 5

DATA_DIR = "data/"
FAISS_INDEX_PATH = "faiss_index"

# LangSmith tracing — set these so every chain.invoke() is logged automatically
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "production-rag-pipeline"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
