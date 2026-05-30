import json
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from agent_router import AgentRAGWorkflow
from config import (
    CHUNKS_PATH,
    DOCUMENT_CATEGORIES,
    FAISS_INDEX_PATH,
    INDEX_MANIFEST_PATH,
    FINAL_TOP_K,
    INDEX_META_PATH,
    SIMILARITY_THRESHOLD,
    USE_HYBRID_SEARCH,
    USE_RERANK,
    VECTOR_TOP_K,
)
from service_context import env_snapshot, log_event, timed_stage


DEFAULT_CATEGORY = "all"


def _safe_int(value: Optional[int], default: int, minimum: int = 1, maximum: int = 50) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))


def _safe_float(value: Optional[float], default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value is None:
        return default
    return max(minimum, min(maximum, float(value)))


def _load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _cached_index_meta() -> Dict:
    with timed_stage("load chunks / metadata cost", file=INDEX_META_PATH):
        return _load_json_file(INDEX_META_PATH, default={})


@lru_cache(maxsize=1)
def _cached_chunks() -> List[Dict]:
    with timed_stage("load chunks / metadata cost", file=CHUNKS_PATH):
        return _load_json_file(CHUNKS_PATH, default=[])


def _compact_source(doc: Dict) -> Dict:
    return {
        "source": doc.get("source", ""),
        "doc_id": doc.get("doc_id"),
        "chunk_id": doc.get("chunk_id", ""),
        "matched_child_id": doc.get("matched_child_id"),
        "chunk_uid": doc.get("chunk_uid"),
        "vector_id": doc.get("vector_id"),
        "paper_title": doc.get("paper_title"),
        "section_title": doc.get("section_title"),
        "section_type": doc.get("section_type"),
        "page_start": doc.get("page_start"),
        "page_end": doc.get("page_end"),
        "parent_id": doc.get("parent_id"),
        "context_mode": doc.get("context_mode"),
        "category": doc.get("category", "general"),
        "relative_path": doc.get("relative_path", ""),
        "path": doc.get("path", ""),
        "extension": doc.get("extension", ""),
        "vector_score": doc.get("vector_score"),
        "bm25_score": doc.get("bm25_score"),
        "rerank_score": doc.get("rerank_score"),
    }


def _compact_chunk(doc: Dict, include_text: bool = True, max_chars: int = 1200) -> Dict:
    item = _compact_source(doc)
    item["chunk_length"] = doc.get("chunk_length")
    item["context_length"] = doc.get("context_length")
    item["chunk_token_count"] = doc.get("chunk_token_count")
    item["parent_token_count"] = doc.get("parent_token_count")
    if include_text:
        text = doc.get("text", "")
        item["text"] = text[:max_chars]
        item["text_truncated"] = len(text) > max_chars
        if doc.get("retrieval_text") and doc.get("retrieval_text") != text:
            item["retrieval_text"] = doc["retrieval_text"][:max_chars]
    return item


@lru_cache(maxsize=4)
def get_rag_pipeline(use_rerank: bool = USE_RERANK):
    """
    Lazily create and cache RAGPipeline instances.

    This is important for stdio MCP servers because the server process should
    stay lightweight at startup and only load models when a tool is called.
    """
    with timed_stage("load config cost", use_rerank=use_rerank):
        from rag_pipline import RAGPipeline

    log_event("model cache environment", **env_snapshot())
    with timed_stage("create RAGPipeline cost", use_rerank=use_rerank):
        return RAGPipeline(use_rerank=use_rerank)


@lru_cache(maxsize=4)
def get_cached_agent_workflow(use_rerank: bool = USE_RERANK) -> AgentRAGWorkflow:
    with timed_stage("create AgentRAGWorkflow cost", use_rerank=use_rerank):
        return AgentRAGWorkflow(get_rag_pipeline(use_rerank=use_rerank))


def get_index_status() -> Dict:
    meta = _cached_index_meta()
    chunks = _cached_chunks()

    return {
        "faiss_index_exists": os.path.exists(FAISS_INDEX_PATH),
        "chunks_exists": os.path.exists(CHUNKS_PATH),
        "index_meta_exists": os.path.exists(INDEX_META_PATH),
        "index_manifest_exists": os.path.exists(INDEX_MANIFEST_PATH),
        "chunk_count": len(chunks),
        "document_count": meta.get("document_count"),
        "embedding_model": meta.get("embedding_model"),
        "embedding_dimension": meta.get("embedding_dimension"),
        "embedding_batch_size": meta.get("embedding_batch_size"),
        "embedding_max_seq_length": meta.get("embedding_max_seq_length"),
        "faiss_index_type": meta.get("faiss_index_type"),
        "chunk_strategy": meta.get("chunk_strategy"),
        "chunk_token_size": meta.get("chunk_token_size"),
        "chunk_token_overlap": meta.get("chunk_token_overlap"),
        "parent_token_size": meta.get("parent_token_size"),
        "parent_token_overlap": meta.get("parent_token_overlap"),
        "manifest_path": meta.get("manifest_path"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "documents": meta.get("documents", []),
    }


def list_knowledge_sources() -> Dict:
    meta = _cached_index_meta()
    documents = meta.get("documents", [])

    if not documents:
        chunks = _cached_chunks()
        seen = {}
        for chunk in chunks:
            source = chunk.get("source", "")
            if not source:
                continue
            key = chunk.get("relative_path", source)
            seen.setdefault(
                key,
                {
                    "source": source,
                    "doc_id": chunk.get("doc_id"),
                    "relative_path": key,
                    "category": chunk.get("category", DOCUMENT_CATEGORIES.get(source, "general")),
                    "path": chunk.get("path", ""),
                    "extension": chunk.get("extension", ""),
                    "chunk_count": 0,
                },
            )
            seen[key]["chunk_count"] += 1
        documents = list(seen.values())

    return {
        "source_count": len(documents),
        "sources": documents,
        "category_options": ["all", *sorted(set(DOCUMENT_CATEGORIES.values()))],
    }


def retrieve_security_chunks(
    query: str,
    category: str = DEFAULT_CATEGORY,
    top_k: int = VECTOR_TOP_K,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = False,
    include_text: bool = True,
    max_chars: int = 1200,
) -> Dict:
    request_start = time.perf_counter()
    rag = get_rag_pipeline(use_rerank=use_rerank)
    top_k = _safe_int(top_k, VECTOR_TOP_K)

    with timed_stage("retrieve candidates cost", query=query, use_hybrid=use_hybrid, top_k=top_k):
        if use_hybrid:
            candidates = rag.hybrid_retrieve(
                query=query,
                vector_top_k=top_k,
                category=category,
            )
        else:
            candidates = rag.vector_retrieve(
                query=query,
                top_k=top_k,
                category=category,
            )

    with timed_stage("rerank cost", query=query, candidate_count=len(candidates), final_top_k=top_k):
        docs = rag.rerank(query, candidates, final_top_k=top_k)
    best_score = max([doc.get("vector_score", 0.0) for doc in candidates], default=0.0)

    return {
        "query": query,
        "category": category,
        "use_hybrid": use_hybrid,
        "use_rerank": rag.use_rerank,
        "top_k": top_k,
        "best_score": best_score,
        "elapsed_seconds": round(time.perf_counter() - request_start, 4),
        "chunks": [
            _compact_chunk(doc, include_text=include_text, max_chars=max_chars)
            for doc in docs
        ],
    }


def answer_security_question(
    query: str,
    category: str = DEFAULT_CATEGORY,
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = USE_RERANK,
    save_log: bool = True,
) -> Dict:
    rag = get_rag_pipeline(use_rerank=use_rerank)
    result = rag.answer(
        query=query,
        category=category,
        vector_top_k=_safe_int(vector_top_k, VECTOR_TOP_K),
        final_top_k=_safe_int(final_top_k, FINAL_TOP_K),
        threshold=_safe_float(threshold, SIMILARITY_THRESHOLD),
        use_hybrid=use_hybrid,
        save_log=save_log,
    )

    return {
        "query": result["query"],
        "answer": result["answer"],
        "status": result["status"],
        "best_score": result["best_score"],
        "use_hybrid": result.get("used_hybrid", False),
        "use_rerank": result.get("used_rerank", False),
        "sources": [_compact_source(doc) for doc in result.get("retrieved_docs", [])],
    }


def agent_security_answer(
    query: str,
    category: str = DEFAULT_CATEGORY,
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = USE_RERANK,
    save_log: bool = True,
) -> Dict:
    agent = get_cached_agent_workflow(use_rerank=use_rerank)
    result = agent.agent_answer(
        query=query,
        category=category,
        vector_top_k=_safe_int(vector_top_k, VECTOR_TOP_K),
        final_top_k=_safe_int(final_top_k, FINAL_TOP_K),
        threshold=_safe_float(threshold, SIMILARITY_THRESHOLD),
        use_hybrid=use_hybrid,
        save_log=save_log,
    )

    return {
        "query": result["query"],
        "task_type": result["task_type"],
        "tool_used": result["tool_used"],
        "answer": result["answer"],
        "status": result["status"],
        "best_score": result["best_score"],
        "use_hybrid": result.get("used_hybrid", False),
        "use_rerank": result.get("used_rerank", False),
        "sources": result["sources"],
    }


def warmup_rag(
    use_rerank: bool = False,
    load_agent: bool = False,
) -> Dict:
    start = time.perf_counter()
    rag = get_rag_pipeline(use_rerank=use_rerank)
    agent_loaded = False
    if load_agent:
        get_cached_agent_workflow(use_rerank=use_rerank)
        agent_loaded = True

    return {
        "status": "warmed",
        "use_rerank": rag.use_rerank,
        "agent_loaded": agent_loaded,
        "elapsed_seconds": round(time.perf_counter() - start, 4),
        "chunk_count": len(rag.chunks or []),
        "env": env_snapshot(),
    }


def benchmark_retrieval(
    query: str = "SQL injection prevention",
    category: str = DEFAULT_CATEGORY,
    top_k: int = 2,
    use_hybrid: bool = USE_HYBRID_SEARCH,
    use_rerank: bool = False,
) -> Dict:
    first_start = time.perf_counter()
    first = retrieve_security_chunks(
        query=query,
        category=category,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        include_text=False,
    )
    first_cost = time.perf_counter() - first_start

    second_start = time.perf_counter()
    second = retrieve_security_chunks(
        query=query,
        category=category,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        include_text=False,
    )
    second_cost = time.perf_counter() - second_start

    return {
        "query": query,
        "first_call_seconds": round(first_cost, 4),
        "second_call_seconds": round(second_cost, 4),
        "speedup": round(first_cost / second_cost, 2) if second_cost > 0 else None,
        "first_best_score": first["best_score"],
        "second_best_score": second["best_score"],
        "chunk_count": len(second["chunks"]),
    }
