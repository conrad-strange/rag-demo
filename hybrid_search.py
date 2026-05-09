import re
from typing import List, Dict

from rank_bm25 import BM25Okapi


def simple_tokenize(text: str):
    """
    分词函数。
    - 中文：用 jieba 分词，保留词组语义
    - 英文：按单词、数字、连字符切分
    - 混合文本：先用正则分离中英文区域，分别处理
    """
    text = text.lower()
    tokens = []
    
    # 用正则将文本按"中文连续块"和"非中文块"交替拆分
    # 例如 "hello机器学习world" -> ["hello", "机器学习", "world"]
    segments = re.split(r'([\u4e00-\u9fff]+)', text)
    
    for seg in segments:
        if not seg:
            continue
        
        # 中文字段：用 jieba 分词
        if re.match(r'[\u4e00-\u9fff]+', seg):
            import jieba
            tokens.extend(jieba.lcut(seg))
        else:
            # 非中文字段：英文/数字/符号，按原逻辑分词
            tokens.extend(re.findall(r"[a-zA-Z0-9_\-]+", seg))
    return tokens


class BM25Retriever:
    """
    基于 BM25 的关键词检索器。
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.corpus_tokens = [
            simple_tokenize(chunk["text"])
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 8, category: str = "all") -> List[Dict]:
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        candidates = []

        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]

            if category != "all" and chunk.get("category") != category:
                continue

            item = chunk.copy()
            item["bm25_score"] = float(score)
            candidates.append(item)

        candidates = sorted(
            candidates,
            key=lambda x: x["bm25_score"],
            reverse=True
        )

        return candidates[:top_k]