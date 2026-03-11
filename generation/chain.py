from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.langchain import LangChainTracer
from generation.llm import get_llm
from generation.prompt import RAG_PROMPT


def format_context(docs):
    return "\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def build_rag_chain(retriever):
    llm = get_llm()
    tracer = LangChainTracer(project_name="production-rag-pipeline")

    chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ).with_config({"callbacks": [tracer]})

    return chain
