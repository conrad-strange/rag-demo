import streamlit as st

from agent_router import AgentRAGWorkflow
from config import (
    EMBEDDING_MODEL_NAME,
    FINAL_TOP_K,
    SIMILARITY_THRESHOLD,
    USE_RERANK,
    VECTOR_TOP_K,
)
from rag_pipline import RAGPipeline


@st.cache_resource
def load_rag_pipeline(use_rerank: bool):
    return RAGPipeline(use_rerank=use_rerank)


def render_sources(docs):
    st.subheader("Retrieved Sources")
    for i, doc in enumerate(docs, start=1):
        title = (
            f"Top {i} | {doc.get('source', '')} | chunk {doc.get('chunk_id', '')} | "
            f"vector_score {doc.get('vector_score', 0.0):.4f}"
        )
        if doc.get("bm25_score") is not None:
            title += f" | bm25_score {doc['bm25_score']:.4f}"
        if doc.get("rerank_score") is not None:
            title += f" | rerank_score {doc['rerank_score']:.4f}"

        with st.expander(title):
            st.write(doc.get("text", ""))


def main():
    st.set_page_config(
        page_title="Security RAG Assistant v2",
        page_icon="RAG",
        layout="wide",
    )

    st.title("Security RAG Assistant v2")
    st.write(
        "A lightweight security RAG demo with vector retrieval, optional hybrid search, "
        "reranking, and an Agent RAG workflow."
    )

    with st.sidebar:
        st.header("Settings")

        app_mode = st.radio(
            "Run mode",
            options=["Normal RAG Mode", "Agent RAG Mode"],
            index=0,
        )

        use_rerank = st.checkbox("Use rerank", value=USE_RERANK)

        vector_top_k = st.slider(
            "Vector top-k",
            min_value=3,
            max_value=15,
            value=VECTOR_TOP_K,
        )

        final_top_k = st.slider(
            "Final top-k",
            min_value=1,
            max_value=8,
            value=FINAL_TOP_K,
        )

        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=SIMILARITY_THRESHOLD,
            step=0.05,
        )

        category = st.selectbox(
            "Document category",
            options=["all", "incident_response", "web_security", "llm_security"],
            index=0,
        )

        use_hybrid = st.checkbox("Use hybrid search (BM25 + vector)", value=False)

        st.markdown("---")
        st.caption("Embedding model")
        st.code(EMBEDDING_MODEL_NAME)

    try:
        rag = load_rag_pipeline(use_rerank=use_rerank)
    except Exception as e:
        st.error("Failed to load RAG pipeline. Run `python build_index.py` and check `.env`.")
        st.exception(e)
        return

    examples = [
        "What is SQL injection?",
        "Summarize the main security risks in this document.",
        "Compare SQL injection and XSS.",
        "What fields are included in the table?",
    ]
    query = st.text_input(
        "Ask a security question",
        placeholder=examples[0],
    )

    with st.expander("Example queries"):
        for example in examples:
            st.code(example)

    if st.button("Run", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving context and generating answer..."):
            if app_mode == "Agent RAG Mode":
                agent = AgentRAGWorkflow(rag)
                result = agent.agent_answer(
                    query=query,
                    category=category,
                    vector_top_k=vector_top_k,
                    final_top_k=final_top_k,
                    threshold=threshold,
                    use_hybrid=use_hybrid,
                    save_log=True,
                )
            else:
                result = rag.answer(
                    category=category,
                    query=query,
                    vector_top_k=vector_top_k,
                    final_top_k=final_top_k,
                    threshold=threshold,
                    use_hybrid=use_hybrid,
                    save_log=True,
                )

        st.subheader("Answer")
        if result["status"] == "answered":
            st.success("Answered with retrieved context.")
        elif result["status"] == "table_fallback_to_docs":
            st.info("No table metadata was found. Fell back to general document retrieval.")
        else:
            st.warning("Insufficient retrieved context.")

        st.write(result["answer"])

        st.subheader("Run Details")
        metric_cols = st.columns(5 if app_mode == "Agent RAG Mode" else 4)
        metric_cols[0].metric("Status", result["status"])
        metric_cols[1].metric("Best score", f"{result['best_score']:.4f}")
        metric_cols[2].metric("Rerank", str(result["used_rerank"]))
        metric_cols[3].metric("Hybrid", str(result.get("used_hybrid", False)))

        if app_mode == "Agent RAG Mode":
            metric_cols[4].metric("Task type", result["task_type"])
            st.write("Tool used:", result["tool_used"])
            st.json(
                {
                    "query": result["query"],
                    "task_type": result["task_type"],
                    "tool_used": result["tool_used"],
                    "sources": result["sources"],
                }
            )

        render_sources(result["retrieved_docs"])


if __name__ == "__main__":
    main()
