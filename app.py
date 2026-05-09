import os
import json
import faiss
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


INDEX_DIR = "index"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
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


def main():
    st.set_page_config(
        page_title="网络安全知识库 RAG Demo",
        page_icon="🔐",
        layout="wide"
    )

    st.title("🔐 网络安全知识库 RAG 问答 Demo")
    st.write("基于本地网络安全文档、FAISS 向量检索和 DeepSeek API 的最小 RAG 原型系统。")

    if not DEEPSEEK_API_KEY:
        st.error("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")
        return

    try:
        embedder, index, chunks = load_resources()
    except Exception as e:
        st.error("索引加载失败，请先运行 python build_index.py")
        st.exception(e)
        return

    with st.sidebar:
        st.header("参数设置")

        top_k = st.slider(
            "Top-K 检索数量",
            min_value=1,
            max_value=5,
            value=3
        )

        threshold = st.slider(
            "信息不足判断阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )

        st.markdown("---")
        st.write("知识库信息")
        st.write(f"chunk 数量：{len(chunks)}")
        st.write(f"Embedding 模型：{EMBEDDING_MODEL_NAME}")

    query = st.text_input(
        "请输入你的问题：",
        placeholder="例如：什么是中间人攻击？SQL注入有哪些防御方法？"
    )

    if st.button("提交问题", type="primary"):
        if not query.strip():
            st.warning("请输入问题。")
            return

        with st.spinner("正在检索知识库并生成回答..."):
            result = rag_answer(
                query=query,
                embedder=embedder,
                index=index,
                chunks=chunks,
                top_k=top_k,
                threshold=threshold
            )

        st.subheader("回答")
        if result["status"] == "answered":
            st.success("已根据知识库生成回答")
        else:
            st.warning("知识库信息不足")

        st.write(result["answer"])

        st.subheader("状态信息")
        col1, col2 = st.columns(2)
        col1.metric("状态", result["status"])
        col2.metric("最高相似度", f"{result['best_score']:.4f}")

        st.subheader("Top-K 检索结果")

        for i, doc in enumerate(result["retrieved_docs"], start=1):
            with st.expander(
                f"Top {i} | {doc['source']} | chunk {doc['chunk_id']} | score {doc['score']:.4f}"
            ):
                st.write(doc["text"])


if __name__ == "__main__":
    main()