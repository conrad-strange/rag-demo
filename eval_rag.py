import os
import json
import pandas as pd
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.csv")
RESULT_FILE = os.path.join(OUTPUT_DIR, "eval_results.csv")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def load_resources():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError("没有找到 index/faiss.index，请先运行 python build_index.py")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError("没有找到 index/chunks.json，请先运行 python build_index.py")

    index = faiss.read_index(index_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embedder, index, chunks


def retrieve(query, embedder, index, chunks, top_k=3):
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        item = chunks[idx]
        results.append({
            "score": float(score),
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "chunk_length": item.get("chunk_length", len(item["text"])),
            "text": item["text"]
        })

    return results


def build_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[来源: {doc['source']} | chunk: {doc['chunk_id']} | score: {doc['score']:.4f}]\n{doc['text']}"
        for doc in retrieved_docs
    ])

    prompt = f"""
你是一个严谨的网络安全知识库问答助手。你只能根据【检索资料】回答问题。

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


def ask_deepseek(prompt):
    if not DEEPSEEK_API_KEY:
        raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
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
        temperature=0.2
    )

    return response.choices[0].message.content


def rag_answer(query, embedder, index, chunks, top_k=3, threshold=0.35):
    retrieved_docs = retrieve(
        query=query,
        embedder=embedder,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    best_score = retrieved_docs[0]["score"] if retrieved_docs else 0

    if best_score < threshold:
        return {
            "query": query,
            "answer": "知识库中没有足够信息回答该问题。",
            "retrieved_docs": retrieved_docs,
            "best_score": best_score,
            "status": "insufficient_context"
        }

    prompt = build_prompt(query, retrieved_docs)
    answer = ask_deepseek(prompt)

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "best_score": best_score,
        "status": "answered"
    }


def keyword_hit(answer, expected_keywords):
    """
    判断回答中是否包含预期关键词。
    """
    if pd.isna(expected_keywords) or str(expected_keywords).strip() == "":
        return {
            "keywords": [],
            "hit_keywords": [],
            "hit_rate": None
        }

    keywords = [
        kw.strip()
        for kw in str(expected_keywords).split(";")
        if kw.strip()
    ]

    if not keywords:
        return {
            "keywords": [],
            "hit_keywords": [],
            "hit_rate": None
        }

    hit_keywords = [kw for kw in keywords if kw in answer]
    hit_rate = len(hit_keywords) / len(keywords)

    return {
        "keywords": keywords,
        "hit_keywords": hit_keywords,
        "hit_rate": hit_rate
    }


def source_hit(retrieved_docs, expected_source):
    """
    判断 Top-K 检索结果中是否包含预期来源文件。
    """
    if pd.isna(expected_source) or expected_source == "none":
        return None

    retrieved_sources = [doc["source"] for doc in retrieved_docs]

    return expected_source in retrieved_sources


def evaluate(top_k=3, threshold=0.35):
    if not os.path.exists(EVAL_FILE):
        raise FileNotFoundError("没有找到 data/eval_questions.csv，请先创建评估问题文件。")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("正在加载索引和模型...")
    embedder, index, chunks = load_resources()

    print("正在读取评估问题...")
    eval_df = pd.read_csv(EVAL_FILE)

    eval_results = []

    for i, row in eval_df.iterrows():
        question = row["question"]
        expected_keywords = row.get("expected_keywords", "")
        expected_source = row.get("expected_source", "none")
        should_answer = row.get("should_answer", "yes")

        print(f"\n[{i + 1}/{len(eval_df)}] 评估问题：{question}")

        result = rag_answer(
            query=question,
            embedder=embedder,
            index=index,
            chunks=chunks,
            top_k=top_k,
            threshold=threshold
        )

        answer = result["answer"]
        retrieved_docs = result["retrieved_docs"]

        kw_result = keyword_hit(answer, expected_keywords)
        src_hit = source_hit(retrieved_docs, expected_source)

        if should_answer == "yes":
            refusal_correct = result["status"] == "answered"
        else:
            refusal_correct = result["status"] == "insufficient_context"

        top1_source = retrieved_docs[0]["source"] if retrieved_docs else ""
        top1_score = retrieved_docs[0]["score"] if retrieved_docs else 0
        top1_chunk_id = retrieved_docs[0]["chunk_id"] if retrieved_docs else ""

        retrieved_sources = ";".join([
            f"{doc['source']}#chunk{doc['chunk_id']}({doc['score']:.4f})"
            for doc in retrieved_docs
        ])

        eval_results.append({
            "question": question,
            "should_answer": should_answer,
            "status": result["status"],
            "best_score": result["best_score"],
            "expected_source": expected_source,
            "top1_source": top1_source,
            "top1_chunk_id": top1_chunk_id,
            "top1_score": top1_score,
            "source_hit": src_hit,
            "expected_keywords": expected_keywords,
            "hit_keywords": ";".join(kw_result["hit_keywords"]),
            "keyword_hit_rate": kw_result["hit_rate"],
            "refusal_correct": refusal_correct,
            "retrieved_sources": retrieved_sources,
            "answer": answer
        })

    result_df = pd.DataFrame(eval_results)

    result_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n评估完成！")
    print(f"结果已保存到：{RESULT_FILE}")

    print_summary(result_df)

    return result_df


def print_summary(result_df):
    print("\n========== 评估摘要 ==========")

    answerable_df = result_df[result_df["should_answer"] == "yes"]
    unanswerable_df = result_df[result_df["should_answer"] == "no"]

    print("总问题数：", len(result_df))
    print("可回答问题数：", len(answerable_df))
    print("不可回答问题数：", len(unanswerable_df))

    if len(answerable_df) > 0:
        print("\n可回答问题：")
        print("平均最高相似度：", round(answerable_df["best_score"].mean(), 4))

        if "source_hit" in answerable_df.columns:
            print("来源命中率：", round(answerable_df["source_hit"].mean(), 4))

        if "keyword_hit_rate" in answerable_df.columns:
            print("平均关键词命中率：", round(answerable_df["keyword_hit_rate"].mean(), 4))

    if len(unanswerable_df) > 0:
        print("\n不可回答问题：")
        print("拒答准确率：", round(unanswerable_df["refusal_correct"].mean(), 4))


if __name__ == "__main__":
    evaluate(
        top_k=3,
        threshold=0.35
    )