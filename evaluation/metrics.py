from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Standard RAGAS metric set — covers both generation quality and retrieval quality
METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]
