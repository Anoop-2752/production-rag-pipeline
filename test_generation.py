from ingestion.document_loader import load_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.vector_store import load_vector_store
from retrieval.retriever import get_retriever
from generation.chain import build_rag_chain

print("\n🤖 Testing Generation...\n")

# Load chunks for hybrid retrieval
docs = load_all_pdfs()
chunks = chunk_documents(docs)

# Load retriever (uses saved FAISS index)
retriever = get_retriever(chunks)

# Build RAG chain
chain = build_rag_chain(retriever)

# Test questions
questions = [
    "What is this document about?",
    "What are the key points discussed?",
    "Summarize the main findings."
]

for question in questions:
    print(f"❓ Question: {question}")
    answer = chain.invoke(question)
    print(f"💬 Answer: {answer}")
    print("-" * 60)