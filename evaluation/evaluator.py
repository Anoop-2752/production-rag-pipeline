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
    Trigram-overlap fallback for when RAGAS faithfulness returns NaN.
    This happens when the LLM doesn't produce valid JSON for statement extraction.
    Scores each answer sentence by checking if any 3-word phrase appears in the context.
    Not a perfect substitute, but beats reporting NaN.
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
    Runs RAGAS evaluation over a set of questions and reference answers.
    Returns a DataFrame with per-question scores and the raw RAGAS results object.
    """
    answers, contexts = [], []

    for question in test_questions:
        answer = rag_chain.invoke(question)
        answers.append(answer)

        docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in docs])

        print(f"  evaluated: {question[:60]}")

    eval_dataset = Dataset.from_dict({
        "user_input": test_questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    results = evaluate(
        dataset=eval_dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
    )
    df = results.to_pandas()

    # Groq sometimes fails the faithfulness JSON extraction — patch with trigram fallback
    if "faithfulness" in df.columns and df["faithfulness"].isna().any():
        print("faithfulness NaN detected, applying trigram fallback")
        for i, row in df.iterrows():
            if pd.isna(row["faithfulness"]):
                df.at[i, "faithfulness"] = _fallback_faithfulness(answers[i], contexts[i])

    return df, results
