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


APP_TITLE = "AI Research Paper RAG Demo"
APP_DESCRIPTION = (
    "A lightweight research-paper RAG demo for AI interpretability, "
    "alignment, deception, trustworthiness, and safety evaluation."
)
EXAMPLE_QUERIES = [
    "Compare alignment faking and Sleeper Agents. What kinds of deception risks do they study?",
    "How do sparse autoencoders help mechanistic interpretability in Towards Monosemanticity and Scaling Monosemanticity?",
]


@st.cache_resource
def load_rag_pipeline(use_rerank: bool):
    return RAGPipeline(use_rerank=use_rerank)


def get_category_options(rag: RAGPipeline):
    categories = sorted(
        {
            chunk.get("category", "general")
            for chunk in (rag.chunks or [])
            if chunk.get("category")
        }
    )
    return ["all", *categories]


def render_sources(docs):
    st.subheader("Retrieved Sources")
    for i, doc in enumerate(docs, start=1):
        title = (
            f"Top {i} | {doc.get('source', '')} | "
            f"{doc.get('section_title') or 'unknown section'} | "
            f"chunk {doc.get('matched_child_id', doc.get('chunk_id', ''))} | "
            f"vector_score {doc.get('vector_score', 0.0):.4f}"
        )
        if doc.get("bm25_score") is not None:
            title += f" | bm25_score {doc['bm25_score']:.4f}"
        if doc.get("rerank_score") is not None:
            title += f" | rerank_score {doc['rerank_score']:.4f}"

        with st.expander(title):
            if doc.get("context_mode"):
                st.caption(f"Context mode: {doc.get('context_mode')} | Parent id: {doc.get('parent_id')}")
            if doc.get("page_start") is not None:
                page_label = str(doc.get("page_start"))
                if doc.get("page_end") not in (None, doc.get("page_start")):
                    page_label = f"{doc.get('page_start')}-{doc.get('page_end')}"
                st.caption(f"Pages: {page_label}")
            st.write(doc.get("text", ""))


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="RAG",
        layout="wide",
    )

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

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

        use_hybrid = st.checkbox("Use hybrid search (BM25 + vector)", value=False)

    try:
        rag = load_rag_pipeline(use_rerank=use_rerank)
    except Exception as e:
        st.error("Failed to load RAG pipeline. Run `python index_manager.py update` and check `.env`.")
        st.exception(e)
        return

    with st.sidebar:
        category = st.selectbox(
            "Corpus category",
            options=get_category_options(rag),
            index=0,
        )

        st.markdown("---")
        st.caption("Embedding model")
        st.code(EMBEDDING_MODEL_NAME)

    query = st.text_input(
        "Ask a research question",
        placeholder=EXAMPLE_QUERIES[0],
    )

    with st.expander("Example queries"):
        for example in EXAMPLE_QUERIES:
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
