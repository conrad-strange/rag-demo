import streamlit as st

from config import (
    VECTOR_TOP_K,
    FINAL_TOP_K,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    USE_RERANK
)
from rag_pipline import RAGPipeline


@st.cache_resource
def load_rag_pipeline(use_rerank: bool):
    return RAGPipeline(use_rerank=use_rerank)


def main():
    st.set_page_config(
        page_title="网络安全知识库 RAG 系统",
        page_icon="🔐",
        layout="wide"
    )

    st.title("🔐 网络安全知识库 RAG 问答系统 v2")
    st.write(
        "基于本地文档、Sentence-Transformers、FAISS、Rerank、Streamlit 和 DeepSeek API 的轻量级 RAG 工具。"
    )

    with st.sidebar:
        st.header("参数设置")

        use_rerank = st.checkbox(
            "启用 Rerank",
            value=USE_RERANK
        )

        vector_top_k = st.slider(
            "第一阶段向量召回数量",
            min_value=3,
            max_value=15,
            value=VECTOR_TOP_K
        )

        final_top_k = st.slider(
            "最终使用的文档片段数量",
            min_value=1,
            max_value=5,
            value=FINAL_TOP_K
        )

        threshold = st.slider(
            "信息不足判断阈值",
            min_value=0.0,
            max_value=1.0,
            value=SIMILARITY_THRESHOLD,
            step=0.05
        )
        category = st.selectbox(
        "知识库范围",
        options=["all", "incident_response", "web_security", "llm_security"],
        index=0
        )
        st.markdown("---")
        st.write("Embedding 模型：")
        st.code(EMBEDDING_MODEL_NAME)

        use_hybrid = st.checkbox(
        "启用 Hybrid Search（BM25 + 向量检索）",
        value=False
        )

    try:
        rag = load_rag_pipeline(use_rerank=use_rerank)
    except Exception as e:
        st.error("RAG 系统加载失败。请确认已经运行 python build_index.py，并检查 .env。")
        st.exception(e)
        return

    query = st.text_input(
        "请输入你的问题：",
        placeholder="例如：什么是中间人攻击？SQL注入如何防御？"
    )

    if st.button("提交问题", type="primary"):
        if not query.strip():
            st.warning("请输入问题。")
            return

        with st.spinner("正在检索知识库并生成回答..."):
            result = rag.answer(
                category=category,
                query=query,
                vector_top_k=vector_top_k,
                final_top_k=final_top_k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                save_log=True
            )

        st.subheader("回答")

        if result["status"] == "answered":
            st.success("已根据知识库生成回答")
        else:
            st.warning("知识库信息不足")

        st.write(result["answer"])

        st.subheader("状态信息")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("状态", result["status"])
        col2.metric("最高向量相似度", f"{result['best_score']:.4f}")
        col3.metric("是否启用 Rerank", str(result["used_rerank"]))
        col4.metric("是否启用 Hybrid", str(result.get("used_hybrid", False)))

        st.subheader("检索结果")

        for i, doc in enumerate(result["retrieved_docs"], start=1):
            title = (
                f"Top {i} | {doc['source']} | chunk {doc['chunk_id']} | "
                f"vector_score {doc.get('vector_score', 0.0):.4f}"
            )

            if doc.get("bm25_score") is not None:
                title += f" | bm25_score {doc['bm25_score']:.4f}"

            if doc.get("rerank_score") is not None:
                title += f" | rerank_score {doc['rerank_score']:.4f}"


if __name__ == "__main__":
    main()