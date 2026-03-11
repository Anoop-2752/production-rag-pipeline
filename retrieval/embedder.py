from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


def get_embeddings():
    """Load HuggingFace embedding model locally — no API cost."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"✅ Embeddings loaded: {EMBEDDING_MODEL}")
    return embeddings