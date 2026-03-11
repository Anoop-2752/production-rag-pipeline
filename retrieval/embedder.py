from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


def get_embeddings():
    """
    Runs the embedding model locally on CPU — no external API calls.
    normalize_embeddings=True ensures cosine similarity works correctly with FAISS.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings
