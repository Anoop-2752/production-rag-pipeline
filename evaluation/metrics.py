from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# These 4 metrics are the industry standard for RAG evaluation
METRICS = [
    faithfulness,        # Is the answer grounded in the context?
    answer_relevancy,    # Is the answer relevant to the question?
    context_precision,   # Are retrieved chunks actually useful?
    context_recall,      # Did we retrieve all necessary information?
]