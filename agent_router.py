import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

from config import FINAL_TOP_K, LOG_DIR, SIMILARITY_THRESHOLD, VECTOR_TOP_K


AGENT_LOG_PATH = os.path.join(LOG_DIR, "agent_log.jsonl")

TASK_TOOL_MAP = {
    "fact_qa": "search_docs",
    "table_qa": "search_tables",
    "summary": "summarize_docs",
    "compare": "compare_docs",
}

ROUTER_RULES = {
    "compare": [
        r"\bcompare\b",
        r"\bdifference(s)?\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bcontrast\b",
        r"对比",
        r"比较",
        r"区别",
        r"差异",
    ],
    "summary": [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\bmain points?\b",
        r"\bkey risks?\b",
        r"总结",
        r"概括",
        r"归纳",
        r"主要.*风险",
    ],
    "table_qa": [
        r"\btable\b",
        r"\bfield(s)?\b",
        r"\bcolumn(s)?\b",
        r"\brow(s)?\b",
        r"\bschema\b",
        r"\bmetadata\b",
        r"表格",
        r"字段",
        r"列",
        r"行",
    ],
}

PROMPT_TEMPLATES = {
    "fact_qa": """You are a security RAG assistant.
Answer the user question only from the retrieved context.
If the context is not enough, say that the retrieved documents do not contain enough evidence.
Keep the answer concise and include source hints when useful.

Context:
{context}

Question:
{query}

Answer:""",
    "table_qa": """You are a security RAG assistant answering a table or field question.
Prefer table-like chunks, field names, column labels, bullet lists, or structured metadata from the context.
If no table evidence is available, clearly say you are falling back to general document context.
Return a compact structured answer.

Context:
{context}

Question:
{query}

Answer:""",
    "summary": """You are a security RAG assistant.
Summarize the retrieved context for the user question.
Use clear bullet points, group similar ideas, and avoid unsupported claims.

Context:
{context}

Question:
{query}

Summary:""",
    "compare": """You are a security RAG assistant.
Compare the requested items using the same dimensions where possible.
Use a clear structure:
1. Similarities
2. Differences
3. Practical security impact
Base the comparison only on the retrieved context.

Context:
{context}

Question:
{query}

Comparison:""",
}


def route_query(query: str) -> str:
    normalized = query.lower().strip()
    for task_type in ("compare", "summary", "table_qa"):
        for pattern in ROUTER_RULES[task_type]:
            if re.search(pattern, normalized):
                return task_type
    return "fact_qa"


def _format_context(docs: List[Dict]) -> str:
    context_parts = []
    for doc in docs:
        score_parts = [f"vector_score={doc.get('vector_score', 0.0):.4f}"]
        if doc.get("bm25_score") is not None:
            score_parts.append(f"bm25_score={doc['bm25_score']:.4f}")
        if doc.get("rerank_score") is not None:
            score_parts.append(f"rerank_score={doc['rerank_score']:.4f}")

        context_parts.append(
            "[source: {source} | chunk: {chunk_id} | {scores}]\n{text}".format(
                source=doc.get("source", ""),
                chunk_id=doc.get("chunk_id", ""),
                scores=", ".join(score_parts),
                text=doc.get("text", ""),
            )
        )
    return "\n\n".join(context_parts)


def _build_prompt(task_type: str, query: str, docs: List[Dict]) -> str:
    template = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES["fact_qa"])
    return template.format(context=_format_context(docs), query=query)


def _source_items(docs: List[Dict]) -> List[Dict]:
    return [
        {
            "source": doc.get("source", ""),
            "chunk_id": doc.get("chunk_id", ""),
            "category": doc.get("category", "general"),
            "vector_score": doc.get("vector_score"),
            "bm25_score": doc.get("bm25_score"),
            "rerank_score": doc.get("rerank_score"),
        }
        for doc in docs
    ]


def _is_table_like(doc: Dict) -> bool:
    metadata_values = [
        str(doc.get("type", "")),
        str(doc.get("content_type", "")),
        str(doc.get("chunk_type", "")),
        str(doc.get("metadata", "")),
    ]
    if any("table" in value.lower() for value in metadata_values):
        return True

    text = doc.get("text", "").lower()
    table_markers = ["|", "\t", "column", "field", "字段", "表格", "列名"]
    return any(marker in text for marker in table_markers)


class AgentRAGWorkflow:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline

    def _retrieve_docs(
        self,
        query: str,
        vector_top_k: int,
        final_top_k: int,
        category: str,
        use_hybrid: bool,
    ) -> Tuple[List[Dict], float]:
        if use_hybrid:
            candidates = self.rag.hybrid_retrieve(
                query=query,
                vector_top_k=vector_top_k,
                category=category,
            )
        else:
            candidates = self.rag.vector_retrieve(
                query=query,
                top_k=vector_top_k,
                category=category,
            )

        best_score = max(
            [doc.get("vector_score", 0.0) for doc in candidates],
            default=0.0,
        )
        return self.rag.rerank(query, candidates, final_top_k=final_top_k), best_score

    def _generate(self, task_type: str, query: str, docs: List[Dict]) -> str:
        prompt = _build_prompt(task_type, query, docs)
        return self.rag.ask_llm(prompt)

    def search_docs(self, query: str, **kwargs) -> Tuple[str, List[Dict], float, str]:
        docs, best_score = self._retrieve_docs(query=query, **kwargs)
        return self._generate("fact_qa", query, docs), docs, best_score, "answered"

    def search_tables(self, query: str, **kwargs) -> Tuple[str, List[Dict], float, str]:
        expanded_kwargs = kwargs.copy()
        expanded_kwargs["vector_top_k"] = max(kwargs["vector_top_k"], 12)
        docs, best_score = self._retrieve_docs(query=query, **expanded_kwargs)
        table_docs = [doc for doc in docs if _is_table_like(doc)]

        if table_docs:
            docs = table_docs[: kwargs["final_top_k"]]
            return self._generate("table_qa", query, docs), docs, best_score, "answered"

        return self._generate("table_qa", query, docs), docs, best_score, "table_fallback_to_docs"

    def summarize_docs(self, query: str, **kwargs) -> Tuple[str, List[Dict], float, str]:
        summary_kwargs = kwargs.copy()
        summary_kwargs["vector_top_k"] = max(kwargs["vector_top_k"], 12)
        summary_kwargs["final_top_k"] = max(kwargs["final_top_k"], 5)
        docs, best_score = self._retrieve_docs(query=query, **summary_kwargs)
        return self._generate("summary", query, docs), docs, best_score, "answered"

    def compare_docs(self, query: str, **kwargs) -> Tuple[str, List[Dict], float, str]:
        compare_kwargs = kwargs.copy()
        compare_kwargs["vector_top_k"] = max(kwargs["vector_top_k"], 12)
        compare_kwargs["final_top_k"] = max(kwargs["final_top_k"], 5)
        docs, best_score = self._retrieve_docs(query=query, **compare_kwargs)
        return self._generate("compare", query, docs), docs, best_score, "answered"

    def agent_answer(
        self,
        query: str,
        category: str = "all",
        vector_top_k: int = VECTOR_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        use_hybrid: bool = False,
        save_log: bool = True,
    ) -> Dict:
        task_type = route_query(query)
        tool_name = TASK_TOOL_MAP[task_type]
        tool = getattr(self, tool_name)

        answer, docs, best_score, status = tool(
            query,
            vector_top_k=vector_top_k,
            final_top_k=final_top_k,
            category=category,
            use_hybrid=use_hybrid,
        )

        if best_score < threshold:
            status = "insufficient_context"
            answer = (
                "The retrieved documents do not contain enough evidence to answer "
                "this question reliably."
            )

        result = {
            "query": query,
            "task_type": task_type,
            "tool_used": tool_name,
            "answer": answer,
            "sources": _source_items(docs),
            "retrieved_docs": docs,
            "status": status,
            "best_score": best_score,
            "used_hybrid": use_hybrid,
            "used_rerank": self.rag.use_rerank,
        }

        if save_log:
            self.save_agent_log(result)

        print(
            "[agent] query={query} task_type={task_type} tool={tool} "
            "sources={sources} answer_len={answer_len} status={status}".format(
                query=query,
                task_type=task_type,
                tool=tool_name,
                sources=[item["source"] for item in result["sources"]],
                answer_len=len(answer or ""),
                status=status,
            )
        )
        return result

    def save_agent_log(self, result: Dict) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": result["query"],
            "task_type": result["task_type"],
            "tool_used": result["tool_used"],
            "status": result["status"],
            "best_score": result["best_score"],
            "answer_length": len(result.get("answer") or ""),
            "sources": result["sources"],
        }
        with open(AGENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def agent_answer(rag_pipeline, query: str, **kwargs) -> Dict:
    return AgentRAGWorkflow(rag_pipeline).agent_answer(query, **kwargs)
