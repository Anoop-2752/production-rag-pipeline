from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    # Model is ~400MB — cache it so it only loads once per process.
    # normalize_embeddings=True ensures cosine similarity works correctly with FAISS.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
