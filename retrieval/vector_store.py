import os
from langchain_community.vectorstores import FAISS
from retrieval.embedder import get_embeddings
from config import FAISS_INDEX_PATH


def build_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to {FAISS_INDEX_PATH}/")
    return vector_store


def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"No FAISS index at '{FAISS_INDEX_PATH}' — run build_vector_store() first."
        )
    embeddings = get_embeddings()
    # allow_dangerous_deserialization needed for FAISS pickle format
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store
