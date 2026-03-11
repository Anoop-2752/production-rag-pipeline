from ingestion.document_loader import load_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.vector_store import build_vector_store
from retrieval.retriever import get_retriever

print("\n🔍 Testing Retrieval...\n")

# Load and chunk
docs = load_all_pdfs()
chunks = chunk_documents(docs)

# Build vector store
build_vector_store(chunks)

# Get hybrid retriever
retriever = get_retriever(chunks)

# Test query
query = "What is this document about?"
results = retriever.invoke(query)

print(f"\n📄 Top results for: '{query}'\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content[:200])
    print()