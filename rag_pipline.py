import os
import json
from datetime import datetime
from typing import List, Dict, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from hybrid_search import BM25Retriever
from config import BM25_TOP_K

from config import (
    ensure_dirs,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    QUERY_LOG_PATH,
    EMBEDDING_MODEL_NAME,
    USE_RERANK,
    RERANK_MODEL_NAME,
    VECTOR_TOP_K,
    FINAL_TOP_K,
    SIMILARITY_THRESHOLD,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_NAME,
    LLM_TEMPERATURE
)


class RAGPipeline:
    """
    轻量级 RAG Pipeline。
    """

    def __init__(
        self,
        use_rerank: bool = USE_RERANK
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

        self._load_resources()
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

        print("正在加载 embedding 模型...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("正在加载 FAISS 索引...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("正在加载 chunks...")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print("正在初始化 BM25 检索器...")
        self.bm25_retriever = BM25Retriever(self.chunks)


    def _load_llm_client(self):
        """
        加载 DeepSeek API Client。
        """
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

            print(f"正在加载 reranker：{RERANK_MODEL_NAME}")
            self.reranker = FlagReranker(
                RERANK_MODEL_NAME,
                use_fp16=False
            )
            print("reranker 加载完成。")

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
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype("float32")

        # 数据量不大时，直接搜索全部 chunks，避免先全局 Top-K 再过滤导致漏召回
        search_k = len(self.chunks)

        scores, indices = self.index.search(query_embedding, search_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            item = self.chunks[idx]

            if category != "all" and item.get("category") != category:
                continue

            results.append({
                "source": item["source"],
                "path": item.get("path", ""),
                "extension": item.get("extension", ""),
                "category": item.get("category", "general"),
                "chunk_id": item["chunk_id"],
                "chunk_length": item.get("chunk_length", len(item["text"])),
                "text": item["text"],
                "vector_score": float(score),
                "rerank_score": None
            })

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
            key = (item["source"], item["chunk_id"])
            item = item.copy()
            item["bm25_score"] = item.get("bm25_score", 0.0)
            merged[key] = item

        for item in bm25_results:
            key = (item["source"], item["chunk_id"])

            if key in merged:
                merged[key]["bm25_score"] = item.get("bm25_score", 0.0)
            else:
                merged[key] = {
                    "source": item["source"],
                    "path": item.get("path", ""),
                    "extension": item.get("extension", ""),
                    "category": item.get("category", "general"),
                    "chunk_id": item["chunk_id"],
                    "chunk_length": item.get("chunk_length", len(item["text"])),
                    "text": item["text"],
                    "vector_score": 0.0,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "rerank_score": None
                }

        return list(merged.values())

    def rerank(self, query: str, candidates: List[Dict], final_top_k: int = FINAL_TOP_K) -> List[Dict]:
        """
        第二阶段：rerank 精排。
        rerank 的作用是对向量召回结果重新排序。
        """
        if not self.use_rerank or self.reranker is None:
            return candidates[:final_top_k]

        pairs = [[query, item["text"]] for item in candidates]

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

        return reranked[:final_top_k]

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
        """
        构造 RAG Prompt。
        """
        context_parts = []

        for doc in retrieved_docs:
            rerank_info = ""
            if doc.get("rerank_score") is not None:
                rerank_info = f" | rerank_score: {doc['rerank_score']:.4f}"
            bm25_info = ""
            if doc.get("bm25_score") is not None:
                bm25_info = f" | bm25_score: {doc['bm25_score']:.4f}"

            context_parts.append(
                f"[来源: {doc['source']} | chunk: {doc['chunk_id']} | "
                f"vector_score: {doc['vector_score']:.4f}{rerank_info}]\n"
                f"{doc['text']}"
                f"vector_score: {doc['vector_score']:.4f}{bm25_info}{rerank_info}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
    你是一个严谨的网络安全知识库问答助手。你只能根据【检索资料】回答用户问题。
    6. If the user's question is in English, answer in English. If the user's question is in Chinese, answer in Chinese.
    请遵守以下规则：
    1. 只使用【检索资料】中的信息回答；
    2. 如果资料不足以回答，请直接说“知识库中没有足够信息回答该问题”；
    3. 不要补充资料中没有出现的事实、案例、工具或数据；
    4. 回答要简洁、分点；
    5. 最后列出你参考的来源文件名和 chunk_id。

    【检索资料】
    {context}

    【用户问题】
    {query}

    【回答】
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
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
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