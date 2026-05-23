import os
import sys
from contextlib import redirect_stdout
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from rag_service import (
    agent_security_answer,
    answer_security_question,
    benchmark_retrieval,
    get_index_status,
    list_knowledge_sources,
    retrieve_security_chunks,
    warmup_rag,
)
from service_context import timed_stage


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


mcp = FastMCP(
    "security-rag-mcp",
    instructions=(
        "Security RAG Assistant MCP server. It exposes local security knowledge "
        "retrieval, normal RAG answering, and Agent RAG answering tools."
    ),
)


def _call_service(func: Callable[..., Any], **kwargs: Any) -> Any:
    # FastMCP stdio uses stdout for protocol frames. Existing RAG code prints
    # model-loading/debug messages, so keep stdout clean and route prints to stderr.
    with redirect_stdout(sys.stderr):
        return func(**kwargs)


def _preload_retrieval_imports() -> None:
    # Windows + FastMCP can be very slow when these scientific dependencies are
    # first imported inside a tool request. Importing them before stdio serving
    # keeps the light default path fast and makes explicit retrieval demos stable.
    with redirect_stdout(sys.stderr):
        with timed_stage("preload retrieval imports cost"):
            import faiss  # noqa: F401
            import numpy  # noqa: F401
            import sentence_transformers  # noqa: F401


@mcp.tool(
    name="get_index_status",
    description="Return local FAISS index, chunk, metadata, and document status for the security RAG knowledge base.",
)
def tool_get_index_status() -> dict:
    return _call_service(get_index_status)


@mcp.tool(
    name="list_knowledge_sources",
    description="List indexed security knowledge sources and document categories.",
)
def tool_list_knowledge_sources() -> dict:
    return _call_service(list_knowledge_sources)


@mcp.tool(
    name="warmup_rag",
    description="Preload the local embedding model and FAISS index so later retrieval calls reuse the cached RAG pipeline.",
)
def tool_warmup_rag(
    use_rerank: bool = False,
    load_agent: bool = False,
) -> dict:
    return _call_service(
        warmup_rag,
        use_rerank=use_rerank,
        load_agent=load_agent,
    )


@mcp.tool(
    name="benchmark_retrieval",
    description="Call retrieve_security_chunks twice in the same server session and report first/second call latency.",
)
def tool_benchmark_retrieval(
    query: str = "SQL injection prevention",
    category: str = "all",
    top_k: int = 2,
    use_hybrid: bool = False,
    use_rerank: bool = False,
) -> dict:
    return _call_service(
        benchmark_retrieval,
        query=query,
        category=category,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )


@mcp.tool(
    name="retrieve_security_chunks",
    description="Retrieve relevant security knowledge chunks without calling the LLM. Useful for grounding and debugging.",
)
def tool_retrieve_security_chunks(
    query: str,
    category: str = "all",
    top_k: int = 8,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    include_text: bool = True,
    max_chars: int = 1200,
) -> dict:
    return _call_service(
        retrieve_security_chunks,
        query=query,
        category=category,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        include_text=include_text,
        max_chars=max_chars,
    )


@mcp.tool(
    name="answer_security_question",
    description="Answer a security question with the normal RAG pipeline and return answer plus sources.",
)
def tool_answer_security_question(
    query: str,
    category: str = "all",
    vector_top_k: int = 8,
    final_top_k: int = 3,
    threshold: float = 0.4,
    use_hybrid: bool = False,
    use_rerank: bool = True,
    save_log: bool = True,
) -> dict:
    return _call_service(
        answer_security_question,
        query=query,
        category=category,
        vector_top_k=vector_top_k,
        final_top_k=final_top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        save_log=save_log,
    )


@mcp.tool(
    name="agent_security_answer",
    description="Answer a security question through Query Router + Tool Selection Agent RAG workflow.",
)
def tool_agent_security_answer(
    query: str,
    category: str = "all",
    vector_top_k: int = 8,
    final_top_k: int = 3,
    threshold: float = 0.4,
    use_hybrid: bool = False,
    use_rerank: bool = True,
    save_log: bool = True,
) -> dict:
    return _call_service(
        agent_security_answer,
        query=query,
        category=category,
        vector_top_k=vector_top_k,
        final_top_k=final_top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        save_log=save_log,
    )


if __name__ == "__main__":
    if "--preload-retrieval-imports" in sys.argv:
        sys.argv.remove("--preload-retrieval-imports")
        _preload_retrieval_imports()

    mcp.run(transport="stdio")
