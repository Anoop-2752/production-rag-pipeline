import re
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from evaluation.metrics import METRICS
from generation.llm import get_llm
from retrieval.embedder import get_embeddings


def _fallback_faithfulness(answer: str, contexts: list) -> float:
    """
    Keyword-overlap faithfulness when RAGAS returns NaN.
    Checks what fraction of answer sentences are supported by the retrieved context.
    """
    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    context_text = " ".join(contexts).lower()
    supported = 0
    for sentence in sentences:
        words = sentence.lower().split()
        if len(words) >= 3:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
            if any(t in context_text for t in trigrams):
                supported += 1
        elif sentence.lower() in context_text:
            supported += 1
    return round(supported / len(sentences), 3)


def run_evaluation(rag_chain, retriever, test_questions: list, ground_truths: list):
    """
    Run RAGAS evaluation on the RAG pipeline.

    Args:
        rag_chain: The built RAG chain
        retriever: The retriever used in the pipeline
        test_questions: List of questions to evaluate
        ground_truths: List of expected answers

    Returns:
        DataFrame with evaluation scores
    """
    print("\n📊 Running RAGAS Evaluation...\n")

    answers = []
    contexts = []

    for question in test_questions:
        # Get answer from RAG chain
        answer = rag_chain.invoke(question)
        answers.append(answer)

        # Get retrieved context
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]
        contexts.append(context)

        print(f"✅ Evaluated: {question[:50]}...")

    # Build evaluation dataset (RAGAS 0.4+ key names)
    eval_dataset = Dataset.from_dict({
        "user_input": test_questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    # Wrap LLM and embeddings for RAGAS
    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # Run evaluation
    results = evaluate(
        dataset=eval_dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
    )

    # Convert to DataFrame
    df = results.to_pandas()

    # Fix NaN faithfulness with keyword-overlap fallback
    if "faithfulness" in df.columns and df["faithfulness"].isna().any():
        print("⚠️  Faithfulness NaN detected — applying fallback scoring...")
        for i, row in df.iterrows():
            if pd.isna(row["faithfulness"]):
                score = _fallback_faithfulness(
                    answers[i], contexts[i]
                )
                df.at[i, "faithfulness"] = score
        print("✅ Fallback faithfulness scores applied")

    return df, results