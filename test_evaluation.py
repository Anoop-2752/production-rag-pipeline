from ingestion.document_loader import load_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.vector_store import load_vector_store
from retrieval.retriever import get_retriever
from generation.chain import build_rag_chain
from evaluation.evaluator import run_evaluation

print("\n📊 Testing Evaluation...\n")

# Load chunks
docs = load_all_pdfs()
chunks = chunk_documents(docs)

# Load retriever and chain
retriever = get_retriever(chunks)
chain = build_rag_chain(retriever)

# ── Define test questions and ground truths ──────────
# Replace these with questions relevant to YOUR document
test_questions = [
    "How does the drowsiness detection module work?",
    "Why was YOLOv8 chosen for object detection?",
    "How does the collision warning system determine danger levels?",
]

ground_truths = [
    "The drowsiness detection module uses MediaPipe Face Mesh to detect 468 facial landmarks, then calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR). If EAR drops below 0.25 for 20 consecutive frames a drowsy alert is triggered, and if MAR exceeds 0.60 for 10 consecutive frames a yawn alert is triggered.",
    "YOLOv8 was chosen because it is one of the fastest object detection models, using a single neural network pass to detect all objects at once. The nano version (yolov8n) runs at 30+ FPS on a regular laptop, which is essential for real-time ADAS applications where speed matters more than perfect accuracy.",
    "The collision warning system calculates the proximity ratio by dividing the nearest vehicle's bounding box area by the total frame area. Below 15% is SAFE, 15-35% is WARNING, and above 35% is DANGER. The state only changes after 5 consecutive frames confirm the new level to prevent flickering.",
]

# Run evaluation
df, results = run_evaluation(chain, retriever, test_questions, ground_truths)

# Print results
print("\n✅ Evaluation Complete!\n")
print("=" * 60)
metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
print(df[["user_input"] + metric_cols].to_string(index=False))
print("=" * 60)
print(f"\n📈 Average Scores:")
for col in metric_cols:
    print(f"  {col}: {df[col].mean():.3f}")

# Save results to CSV
df.to_csv("evaluation_results.csv", index=False)
print("\n💾 Results saved to evaluation_results.csv")