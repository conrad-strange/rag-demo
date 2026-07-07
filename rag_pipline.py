import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
from config import BM25_TOP_K

from config import (
    ensure_dirs,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    INDEX_MANIFEST_PATH,
    QUERY_LOG_PATH,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MAX_SEQ_LENGTH,
    USE_RERANK,
    RERANK_MODEL_NAME,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    SIMILARITY_THRESHOLD,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_NAME,
    LLM_TEMPERATURE
)
from embedding_utils import resolve_embedding_device
from service_context import env_snapshot, log_event, timed_stage


class RAGPipeline:
    """
    轻量级 RAG Pipeline。
    """

    def __init__(
        self,
        use_rerank: bool = USE_RERANK,
        load_llm_client: bool = True,
    ):
        ensure_dirs()
        load_dotenv(override=True)

        self.use_rerank = use_rerank
        self.embedder = None
        self.index = None
        self.chunks = None
        self.reranker = None
        self.client = None
        self.bm25_retriever = None
        self.index_manifest = {}
        self.chunk_by_vector_id = {}

        self._load_resources()
        if load_llm_client:
            self._load_llm_client()

        if self.use_rerank:
            self._load_reranker()

    def _load_resources(self):
        """
        加载 embedding 模型、FAISS 索引和 chunks。
        """
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError("没有找到 faiss.index，请先运行 python build_index.py")

        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError("没有找到 chunks.json，请先运行 python build_index.py")

        log_event(
            "embedding model cache status",
            embedding_model=EMBEDDING_MODEL_NAME,
            **env_snapshot(),
        )
        with timed_stage("load embedding model cost", embedding_model=EMBEDDING_MODEL_NAME):
            from sentence_transformers import SentenceTransformer

            device = resolve_embedding_device(EMBEDDING_DEVICE)
            print(f"Loading embedding model on device: {device}")
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            if EMBEDDING_MAX_SEQ_LENGTH:
                self.embedder.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
                try:
                    self.embedder[0].max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
                except Exception:
                    pass
            log_event(
                "embedding model loaded",
                embedding_model=EMBEDDING_MODEL_NAME,
                embedding_device=device,
                model_path=getattr(self.embedder, "model_name_or_path", EMBEDDING_MODEL_NAME),
            )

        with timed_stage("load faiss index cost", path=FAISS_INDEX_PATH):
            import faiss

            print("Loading FAISS index...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)

        with timed_stage("load chunks / metadata cost", path=CHUNKS_PATH):
            print("Loading chunks...")
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

        if os.path.exists(INDEX_MANIFEST_PATH):
            with timed_stage("load index manifest cost", path=INDEX_MANIFEST_PATH):
                with open(INDEX_MANIFEST_PATH, "r", encoding="utf-8") as f:
                    self.index_manifest = json.load(f)

        self.chunk_by_vector_id = {
            int(chunk["vector_id"]): chunk
            for chunk in self.chunks
            if chunk.get("vector_id") is not None
        }

        with timed_stage("initialize BM25 cost", chunk_count=len(self.chunks)):
            from hybrid_search import BM25Retriever

            print("Initializing BM25 retriever...")
            self.bm25_retriever = BM25Retriever(self.chunks)

    def _chunk_from_faiss_id(self, faiss_id: int) -> Optional[Dict]:
        """
        New indexes use FAISS IndexIDMap2, so search returns vector_id values.
        Older indexes returned positional offsets. Keep both paths readable.
        """
        if faiss_id < 0:
            return None

        if self.chunk_by_vector_id:
            return self.chunk_by_vector_id.get(int(faiss_id))

        if 0 <= faiss_id < len(self.chunks):
            return self.chunks[faiss_id]

        return None

    def _result_from_chunk(
        self,
        item: Dict,
        vector_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
        rerank_score: Optional[float] = None,
    ) -> Dict:
        text = item.get("text", "")
        result = {
            "doc_id": item.get("doc_id"),
            "source": item.get("source", ""),
            "path": item.get("path", ""),
            "relative_path": item.get("relative_path", item.get("source", "")),
            "extension": item.get("extension", ""),
            "category": item.get("category", "general"),
            "paper_title": item.get("paper_title"),
            "chunk_id": item.get("chunk_id"),
            "chunk_uid": item.get("chunk_uid"),
            "vector_id": item.get("vector_id"),
            "chunk_length": item.get("chunk_length", len(text)),
            "chunk_token_count": item.get("chunk_token_count"),
            "chunk_strategy": item.get("chunk_strategy"),
            "section_index": item.get("section_index"),
            "section_title": item.get("section_title"),
            "section_type": item.get("section_type"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "parent_id": item.get("parent_id"),
            "parent_index": item.get("parent_index"),
            "parent_length": item.get("parent_length"),
            "parent_token_count": item.get("parent_token_count"),
            "parent_text": item.get("parent_text"),
            "retrieval_text": item.get("retrieval_text", text),
            "text": text,
            "vector_score": float(vector_score if vector_score is not None else item.get("vector_score", 0.0)),
            "bm25_score": bm25_score if bm25_score is not None else item.get("bm25_score"),
            "rerank_score": rerank_score if rerank_score is not None else item.get("rerank_score"),
        }
        return result

    def _select_parent_contexts(
        self,
        ranked_docs: List[Dict],
        final_top_k: int,
        max_per_source: Optional[int] = None,
    ) -> List[Dict]:
        selected: List[Dict] = []
        seen_contexts = set()
        source_counts: Dict[str, int] = {}
        deferred: List[Dict] = []

        def promote(doc: Dict) -> Dict:
            promoted = doc.copy()
            promoted["retrieval_text"] = doc.get("retrieval_text") or doc.get("text", "")
            promoted["matched_child_id"] = doc.get("chunk_id")
            promoted["matched_child_uid"] = doc.get("chunk_uid")

            if doc.get("parent_text"):
                promoted["text"] = doc["parent_text"]
                promoted["context_mode"] = "parent"
            else:
                promoted["context_mode"] = "child"
            promoted["context_length"] = len(promoted.get("text", ""))
            return promoted

        for doc in ranked_docs:
            context_id = doc.get("parent_id") or doc.get("chunk_uid") or doc.get("vector_id")
            if context_id in seen_contexts:
                continue

            source = doc.get("source", "")
            if max_per_source is not None and source_counts.get(source, 0) >= max_per_source:
                deferred.append(doc)
                continue

            selected.append(promote(doc))
            seen_contexts.add(context_id)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) >= final_top_k:
                return selected

        for doc in deferred:
            context_id = doc.get("parent_id") or doc.get("chunk_uid") or doc.get("vector_id")
            if context_id in seen_contexts:
                continue
            selected.append(promote(doc))
            seen_contexts.add(context_id)
            if len(selected) >= final_top_k:
                break

        return selected


    def _load_llm_client(self):
        """
        加载 DeepSeek API Client。
        """
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

        self.client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL
        )

    def _load_reranker(self):
        """
        加载 reranker。
        如果没有安装 FlagEmbedding，则自动降级为不用 rerank。
        """
        try:
            from FlagEmbedding import FlagReranker

            with timed_stage("load reranker cost", rerank_model=RERANK_MODEL_NAME):
                print(f"Loading reranker: {RERANK_MODEL_NAME}")
                self.reranker = FlagReranker(
                    RERANK_MODEL_NAME,
                    use_fp16=False
                )
            print("reranker loaded.")

        except Exception as e:
            print("reranker 加载失败，将自动关闭 rerank。")
            print("原因：", str(e))
            self.reranker = None
            self.use_rerank = False

    def vector_retrieve(self, query: str, top_k: int = VECTOR_TOP_K, category: str = "all") -> List[Dict]:
        """
        第一阶段：向量召回。
        如果指定 category，则先尽可能召回更多结果，再进行类别过滤。
        """
        import numpy as np

        with timed_stage("first query embedding cost", query=query):
            query_embedding = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )
        query_embedding = np.array(query_embedding).astype("float32")

        # 数据量不大时，直接搜索全部向量，避免先全局 Top-K 再过滤导致漏召回。
        search_k = min(len(self.chunks), int(self.index.ntotal))
        if search_k <= 0:
            return []

        with timed_stage("faiss search cost", search_k=search_k, category=category):
            scores, indices = self.index.search(query_embedding, search_k)

        results = []

        for score, faiss_id in zip(scores[0], indices[0]):
            item = self._chunk_from_faiss_id(int(faiss_id))
            if item is None:
                continue

            if category != "all" and item.get("category") != category:
                continue

            result = self._result_from_chunk(item, vector_score=float(score), rerank_score=None)
            result["vector_id"] = result.get("vector_id") or int(faiss_id)
            results.append(result)

            if len(results) >= top_k:
                break

        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        vector_top_k: int = VECTOR_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        category: str = "all"
    ) -> List[Dict]:
        """
        Hybrid Search:
        1. FAISS 向量召回；
        2. BM25 关键词召回；
        3. 按 source + chunk_id 合并候选。
        """
        vector_results = self.vector_retrieve(
            query=query,
            top_k=vector_top_k,
            category=category
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=bm25_top_k,
            category=category
        )

        merged = {}

        for item in vector_results:
            key = item.get("chunk_uid") or item.get("vector_id") or (item["source"], item["chunk_id"])
            item = item.copy()
            item["bm25_score"] = item.get("bm25_score", 0.0)
            merged[key] = item

        for item in bm25_results:
            key = item.get("chunk_uid") or item.get("vector_id") or (item["source"], item["chunk_id"])

            if key in merged:
                merged[key]["bm25_score"] = item.get("bm25_score", 0.0)
            else:
                merged[key] = self._result_from_chunk(
                    item,
                    vector_score=0.0,
                    bm25_score=item.get("bm25_score", 0.0),
                    rerank_score=None,
                )

        return list(merged.values())

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        final_top_k: int = FINAL_TOP_K,
        max_per_source: Optional[int] = None,
    ) -> List[Dict]:
        """
        第二阶段：rerank 精排。
        rerank 的作用是对向量召回结果重新排序。
        """
        if not candidates:
            return []

        if not self.use_rerank or self.reranker is None:
            log_event(
                "rerank skipped",
                use_rerank=self.use_rerank,
                has_reranker=self.reranker is not None,
                candidate_count=len(candidates),
            )
            return self._select_parent_contexts(
                candidates,
                final_top_k=final_top_k,
                max_per_source=max_per_source,
            )

        pairs = [[query, item["text"]] for item in candidates]

        with timed_stage("rerank cost", candidate_count=len(candidates), final_top_k=final_top_k):
            scores = self.reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        reranked = []

        for item, score in zip(candidates, scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked = sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return self._select_parent_contexts(
            reranked,
            final_top_k=final_top_k,
            max_per_source=max_per_source,
        )

    def retrieve(
        self,
        query: str,
        vector_top_k: int = VECTOR_TOP_K,
        final_top_k: int = FINAL_TOP_K
    ) -> List[Dict]:
        """
        完整检索流程：
        1. 向量召回 top_k；
        2. rerank 精排；
        3. 返回 final_top_k。
        """
        candidates = self.vector_retrieve(query, top_k=vector_top_k)
        final_docs = self.rerank(query, candidates, final_top_k=final_top_k)
        return final_docs

    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Build the final RAG prompt from parent contexts."""
        context_parts = []

        for doc in retrieved_docs:
            score_parts = [f"vector_score: {doc.get('vector_score', 0.0):.4f}"]
            if doc.get("bm25_score") is not None:
                score_parts.append(f"bm25_score: {doc['bm25_score']:.4f}")
            if doc.get("rerank_score") is not None:
                score_parts.append(f"rerank_score: {doc['rerank_score']:.4f}")

            section = doc.get("section_title") or "unknown section"
            page = ""
            if doc.get("page_start") is not None:
                page = f" | page: {doc.get('page_start')}"
                if doc.get("page_end") not in (None, doc.get("page_start")):
                    page = f" | pages: {doc.get('page_start')}-{doc.get('page_end')}"

            context_parts.append(
                f"[source: {doc.get('source', '')} | section: {section}{page} | "
                f"matched_child: {doc.get('matched_child_id', doc.get('chunk_id'))} | "
                f"context: {doc.get('context_mode', 'child')} | {', '.join(score_parts)}]\n"
                f"{doc.get('text', '')}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a careful AI research-paper RAG assistant.
Answer only from the retrieved context. If the context is not enough, say so directly.
If the question is in English, answer in English. If the question is in Chinese, answer in Chinese.
Do not invent facts, paper claims, metrics, datasets, or conclusions that are not supported by the context.
End with the source file names and matched child chunk ids you used.

Retrieved context:
{context}

User question:
{query}

Answer:
"""
        return prompt

    def ask_llm(self, prompt: str) -> str:
        """
        调用 DeepSeek 生成回答。
        """
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的知识库问答助手，只能根据用户提供的检索资料回答。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=LLM_TEMPERATURE
        )

        return response.choices[0].message.content

    def answer(
        self,
        query: str,
        category: str = "all",
        vector_top_k: int = VECTOR_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        use_hybrid: bool = False,
        save_log: bool = True
    ) -> Dict:
        """
        完整 RAG 回答流程。
        """
        if use_hybrid:
            candidates = self.hybrid_retrieve(
                    query=query,
                    vector_top_k=vector_top_k,
                    bm25_top_k=BM25_TOP_K,
                    category=category
            )
        else:
            candidates = self.vector_retrieve(
                    query=query,
                    top_k=vector_top_k,
                    category=category
            )

        best_vector_score = max(
            [doc.get("vector_score", 0.0) for doc in candidates],
            default=0.0
        )

        # 信息不足判断基于向量召回的 best_score。
        # 原因：rerank 分数不是余弦相似度，不能直接和 threshold 比较。
        if best_vector_score < threshold:
            result = {
                "query": query,
                "answer": "知识库中没有足够信息回答该问题。",
                "status": "insufficient_context",
                "used_hybrid": use_hybrid,
                "best_score": best_vector_score,
                "retrieved_docs": candidates[:final_top_k],
                "used_hybrid": use_hybrid,
                "used_rerank": self.use_rerank
            }

            if save_log:
                self.save_query_log(result)

            return result

        retrieved_docs = self.rerank(query, candidates, final_top_k=final_top_k)
        top1_rerank_score = retrieved_docs[0].get("rerank_score")

        if self.use_rerank and top1_rerank_score is not None and top1_rerank_score < -2.0:
            result = {
                "query": query,
                "answer": "知识库中没有足够信息回答该问题。",
                "status": "insufficient_context",
                "best_score": best_vector_score,
                "retrieved_docs": retrieved_docs,
                "used_rerank": self.use_rerank,
                "used_hybrid": use_hybrid
            }
        prompt = self.build_prompt(query, retrieved_docs)
        answer = self.ask_llm(prompt)

        result = {
            "query": query,
            "answer": answer,
            "status": "answered",
            "best_score": best_vector_score,
            "retrieved_docs": retrieved_docs,
            "used_rerank": self.use_rerank
        }

        if save_log:
            self.save_query_log(result)

        return result

    def save_query_log(self, result: Dict):
        """
        保存问答日志到 logs/query_logs.jsonl。
        """
        log_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": result["query"],
            "answer": result["answer"],
            "status": result["status"],
            "best_score": result["best_score"],
            "used_rerank": result["used_rerank"],
            "used_hybrid": result.get("used_hybrid", False),
            "retrieved_docs": [
                {
                    "doc_id": doc.get("doc_id"),
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "chunk_uid": doc.get("chunk_uid"),
                    "vector_id": doc.get("vector_id"),
                    "vector_score": doc["vector_score"],
                    "bm25_score": doc.get("bm25_score"),
                    "rerank_score": doc.get("rerank_score")
                }
                for doc in result["retrieved_docs"]
            ]
        }

        with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rag = RAGPipeline(use_rerank=USE_RERANK)

    question = "What does OWASP say about A03 Injection prevention?"
    result = rag.answer(question)

    print("问题：", question)
    print("状态：", result["status"])
    print("最高向量相似度：", result["best_score"])
    print("回答：")
    print(result["answer"])

    print("\n检索结果：")
    for doc in result["retrieved_docs"]:
        print(doc["source"], doc["chunk_id"], doc["vector_score"], doc.get("rerank_score"))
