from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from retrieval.vector_store import load_vector_store
from config import TOP_K


class EnsembleRetriever(BaseRetriever):
    """Simple hybrid retriever that merges results from multiple retrievers."""
    retrievers: list
    weights: List[float]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        seen = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.page_content
                score = weight * (1 / (rank + 1))
                if key in seen:
                    seen[key] = (seen[key][0] + score, doc)
                else:
                    seen[key] = (score, doc)
        ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]


def get_retriever(chunks=None):
    """
    Hybrid retriever = FAISS (semantic) + BM25 (keyword).
    This is more powerful than pure semantic search alone.
    """
    vector_store = load_vector_store()
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    if chunks is None:
        print("✅ Using semantic retriever only")
        return semantic_retriever

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    print("✅ Hybrid retriever ready (semantic + BM25)")
    return hybrid_retriever