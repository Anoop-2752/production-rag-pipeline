from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from generation.llm import get_llm
from generation.prompt import RAG_PROMPT


def format_context(docs):
    """Format retrieved docs into a single context string."""
    return "\n\n".join([
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])


def build_rag_chain(retriever):
    """
    Build the full RAG chain:
    question → retrieve → format → prompt → LLM → answer
    """
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready")
    return chain