import os
from typing import Any, Dict, Optional

import requests
import streamlit as st


APP_TITLE = "AI Research Paper RAG"
DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "")
REQUEST_TIMEOUT = 180

EXAMPLE_QUERIES = [
    "What is AgentDojo and how does it evaluate prompt injection defenses?",
    "Compare AgentPoison and InjecAgent. What agent security risks do they study?",
    "Summarize the main threat surfaces discussed in recent LLM agent security surveys.",
]


def normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def auth_headers(api_key: str, include_content_type: bool = True) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"} if include_content_type else {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def api_get(base_url: str, path: str, api_key: str = "") -> Dict[str, Any]:
    response = requests.get(
        f"{normalize_base_url(base_url)}{path}",
        headers=auth_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    return handle_response(response)


def api_post(base_url: str, path: str, payload: Dict[str, Any], api_key: str = "") -> Dict[str, Any]:
    response = requests.post(
        f"{normalize_base_url(base_url)}{path}",
        headers=auth_headers(api_key),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    return handle_response(response)


def api_upload(base_url: str, path: str, uploaded_file, api_key: str = "") -> Dict[str, Any]:
    response = requests.post(
        f"{normalize_base_url(base_url)}{path}",
        headers=auth_headers(api_key, include_content_type=False),
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
        timeout=REQUEST_TIMEOUT,
    )
    return handle_response(response)


def handle_response(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}

    if response.status_code >= 400:
        detail = body.get("detail", response.text)
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return body


@st.cache_data(ttl=10)
def cached_index_status(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/index/status", api_key)


@st.cache_data(ttl=30)
def cached_sources(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/sources", api_key)


@st.cache_data(ttl=10)
def cached_documents(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/documents", api_key)


def render_backend_summary(status: Optional[Dict[str, Any]], sources: Optional[Dict[str, Any]]) -> None:
    cols = st.columns(4)
    cols[0].metric("Documents", status.get("document_count", 0) if status else "-")
    cols[1].metric("Chunks", status.get("chunk_count", 0) if status else "-")
    cols[2].metric("Sources", sources.get("source_count", 0) if sources else "-")
    cols[3].metric("Vector dim", status.get("embedding_dimension", 0) if status else "-")


def render_sources(sources):
    st.subheader("Retrieved Sources")
    if not sources:
        st.info("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):
        score = source.get("rerank_score")
        if score is None:
            score = source.get("vector_score")
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "n/a"
        title = (
            f"Top {index} | {source.get('source', '')} | "
            f"{source.get('section_title') or 'unknown section'} | score {score_text}"
        )
        with st.expander(title):
            st.write(
                {
                    "paper_title": source.get("paper_title"),
                    "section_type": source.get("section_type"),
                    "page_start": source.get("page_start"),
                    "page_end": source.get("page_end"),
                    "chunk_id": source.get("chunk_id"),
                    "matched_child_id": source.get("matched_child_id"),
                    "context_mode": source.get("context_mode"),
                    "relative_path": source.get("relative_path"),
                    "vector_score": source.get("vector_score"),
                    "bm25_score": source.get("bm25_score"),
                    "rerank_score": source.get("rerank_score"),
                }
            )


def render_uploaded_documents(documents_info: Optional[Dict[str, Any]]) -> None:
    with st.expander("Uploaded documents"):
        if not documents_info:
            st.info("No upload manifest available.")
            return

        documents = documents_info.get("documents", [])
        if not documents:
            st.info("No uploaded documents yet.")
            return

        for document in documents[:20]:
            st.write(
                {
                    "document_id": document.get("document_id"),
                    "filename": document.get("filename"),
                    "size_bytes": document.get("size_bytes"),
                    "sha256": document.get("sha256"),
                    "status": document.get("status"),
                    "indexed": document.get("indexed"),
                    "created_at": document.get("created_at"),
                    "relative_path": document.get("relative_path"),
                }
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="RAG", layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Backend")
        base_url = st.text_input("API URL", value=DEFAULT_API_URL)
        api_key = st.text_input("API key", value=DEFAULT_API_KEY, type="password")

        st.header("Retrieval")
        mode = st.radio("Mode", ["Normal RAG", "Agent RAG"], horizontal=True)
        category = "all"
        vector_top_k = st.slider("Vector top-k", min_value=3, max_value=20, value=8)
        final_top_k = st.slider("Final top-k", min_value=1, max_value=8, value=3)
        threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        use_hybrid = st.checkbox("Hybrid search", value=False)
        use_rerank = st.checkbox("Rerank", value=True)

        refresh = st.button("Refresh status")
        if refresh:
            cached_index_status.clear()
            cached_sources.clear()
            cached_documents.clear()

        st.header("Documents")
        upload = st.file_uploader("Upload document", type=["pdf", "txt", "md", "docx"])
        if st.button("Upload and index", disabled=upload is None):
            try:
                result = api_upload(base_url, "/documents/upload", upload, api_key)
                cached_documents.clear()
                cached_index_status.clear()
                st.success(f"Queued: {result['filename']}")
            except Exception as exc:
                st.error(str(exc))

    status = None
    sources_info = None
    documents_info = None
    try:
        status = cached_index_status(base_url, api_key)
        sources_info = cached_sources(base_url, api_key)
        documents_info = cached_documents(base_url, api_key)
        categories = sources_info.get("category_options", ["all"])
        if categories:
            category = st.sidebar.selectbox("Category", categories, index=0)
    except Exception as exc:
        st.warning(f"Backend unavailable: {exc}")

    render_backend_summary(status, sources_info)
    render_uploaded_documents(documents_info)

    initial_query = st.session_state.pop("selected_query", "")
    query = st.text_area(
        "Question",
        value=initial_query,
        placeholder=EXAMPLE_QUERIES[0],
        height=110,
    )

    with st.expander("Example queries"):
        for example in EXAMPLE_QUERIES:
            if st.button(example, key=example):
                st.session_state["selected_query"] = example
                st.rerun()

    run = st.button("Run", type="primary", use_container_width=True)
    if not run:
        return

    if not query.strip():
        st.warning("Please enter a question.")
        return

    payload = {
        "query": query.strip(),
        "category": category,
        "vector_top_k": vector_top_k,
        "final_top_k": final_top_k,
        "threshold": threshold,
        "use_hybrid": use_hybrid,
        "use_rerank": use_rerank,
        "save_log": True,
    }
    endpoint = "/agent/chat" if mode == "Agent RAG" else "/chat"

    with st.spinner("Running retrieval and generation through the API..."):
        try:
            result = api_post(base_url, endpoint, payload, api_key)
        except Exception as exc:
            st.error(str(exc))
            return

    st.subheader("Answer")
    if result.get("status") == "answered":
        st.success("Answered")
    else:
        st.warning(result.get("status", "unknown"))
    st.write(result.get("answer", ""))

    cols = st.columns(5 if mode == "Agent RAG" else 4)
    cols[0].metric("Status", result.get("status", "-"))
    cols[1].metric("Best score", f"{result.get('best_score', 0.0):.4f}")
    cols[2].metric("Rerank", str(result.get("use_rerank", False)))
    cols[3].metric("Hybrid", str(result.get("use_hybrid", False)))
    if mode == "Agent RAG":
        cols[4].metric("Task", result.get("task_type", "-"))
        st.caption(f"Tool used: {result.get('tool_used', '-')}")

    render_sources(result.get("sources", []))


if __name__ == "__main__":
    main()
