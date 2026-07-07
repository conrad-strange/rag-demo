import os
import uuid
from html import escape
from typing import Any, Dict, List, Optional

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
def cached_health(base_url: str) -> Dict[str, Any]:
    return api_get(base_url, "/health")


@st.cache_data(ttl=10)
def cached_index_status(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/index/status", api_key)


@st.cache_data(ttl=30)
def cached_sources(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/sources", api_key)


@st.cache_data(ttl=10)
def cached_documents(base_url: str, api_key: str) -> Dict[str, Any]:
    return api_get(base_url, "/documents", api_key)


def clear_api_caches() -> None:
    cached_health.clear()
    cached_index_status.clear()
    cached_sources.clear()
    cached_documents.clear()


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def reset_chat() -> None:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.4rem;
            max-width: 1180px;
        }
        .app-title {
            font-size: 1.55rem;
            line-height: 1.35;
            font-weight: 700;
            margin-bottom: 0.1rem;
            padding-top: 0.15rem;
        }
        .app-subtitle {
            color: #56606d;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.12rem 0.45rem;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid #d5dae1;
            background: #f7f8fa;
            color: #333b45;
            margin-right: 0.25rem;
        }
        .status-indexed {
            border-color: #94c9a6;
            background: #edf8f0;
            color: #225b36;
        }
        .status-indexing, .status-queued {
            border-color: #d4bd76;
            background: #fff8df;
            color: #695214;
        }
        .status-failed {
            border-color: #e4a1a1;
            background: #fff0f0;
            color: #7a2424;
        }
        .doc-row {
            padding: 0.65rem 0;
            border-bottom: 1px solid #edf0f3;
        }
        .doc-row:last-child {
            border-bottom: 0;
        }
        .doc-title {
            font-weight: 600;
            font-size: 0.92rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .doc-meta {
            color: #66707c;
            font-size: 0.78rem;
            margin-top: 0.1rem;
        }
        .source-card {
            border-left: 3px solid #9aa7b5;
            padding-left: 0.75rem;
        }
        .section-heading {
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.35;
            margin: 0 0 0.25rem 0;
        }
        .section-caption {
            color: #66707c;
            font-size: 0.82rem;
            margin: 0 0 0.85rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_class(status: str) -> str:
    normalized = (status or "").lower()
    if normalized in {"indexed", "indexing", "queued", "failed"}:
        return f"status-{normalized}"
    return ""


def render_status_pill(status: str) -> str:
    return f'<span class="status-pill {status_class(status)}">{status or "unknown"}</span>'


def load_backend_state(base_url: str, api_key: str) -> Dict[str, Optional[Dict[str, Any]]]:
    state = {"health": None, "index": None, "sources": None, "documents": None, "error": None}
    try:
        state["health"] = cached_health(base_url)
        state["index"] = cached_index_status(base_url, api_key)
        state["sources"] = cached_sources(base_url, api_key)
        state["documents"] = cached_documents(base_url, api_key)
    except Exception as exc:
        state["error"] = {"detail": str(exc)}
    return state


def render_workspace_summary(state: Dict[str, Optional[Dict[str, Any]]]) -> None:
    if state.get("error"):
        st.warning(f"Backend unavailable: {state['error']['detail']}")
        return

    index = state.get("index") or {}
    sources = state.get("sources") or {}
    documents = state.get("documents") or {}

    cols = st.columns(4)
    cols[0].metric("Indexed docs", index.get("document_count", 0))
    cols[1].metric("Chunks", index.get("chunk_count", 0))
    cols[2].metric("Sources", sources.get("source_count", 0))
    cols[3].metric("Uploads", documents.get("document_count", 0))


def render_document_list(documents_info: Optional[Dict[str, Any]]) -> None:
    st.markdown('<div class="section-heading">Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Uploaded files and indexing state.</div>',
        unsafe_allow_html=True,
    )
    if not documents_info:
        st.info("No document state available.")
        return

    documents = documents_info.get("documents", [])
    if not documents:
        st.caption("No uploaded documents yet.")
        return

    for document in documents[:12]:
        status = document.get("status", "unknown")
        filename = escape(str(document.get("filename", "")))
        size = document.get("size_bytes", 0)
        created = escape(str(document.get("created_at") or "-"))
        st.markdown(
            f"""
            <div class="doc-row">
              <div class="doc-title">{filename}</div>
              <div>{render_status_pill(status)}</div>
              <div class="doc-meta">{size} bytes &middot; {created}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if document.get("error_message"):
            st.caption(f"Error: {document['error_message']}")


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):
        score = source.get("rerank_score")
        if score is None:
            score = source.get("vector_score")
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "n/a"
        title = (
            f"{index}. {source.get('source', '')} · "
            f"{source.get('section_title') or 'unknown section'} · score {score_text}"
        )
        with st.expander(title):
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)


def render_chat_history() -> None:
    st.markdown('<div class="section-heading">Conversation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Ask follow-up questions; memory is scoped to the current session.</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        st.info("Start a research conversation or choose an example from the sidebar.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                meta = message.get("meta", {})
                cols = st.columns(4)
                cols[0].metric("Status", meta.get("status", "-"))
                cols[1].metric("Score", f"{meta.get('best_score', 0.0):.4f}")
                cols[2].metric("Mode", meta.get("mode", "-"))
                cols[3].metric("Message id", meta.get("chat_message_id", "-"))
                if meta.get("memory_used"):
                    st.caption(f"Memory: {meta.get('memory_history_count', 0)} previous turns used")
                if meta.get("task_type"):
                    st.caption(f"Agent task: {meta.get('task_type')} · tool: {meta.get('tool_used', '-')}")
                render_sources(message.get("sources", []))


def append_user_message(content: str) -> None:
    st.session_state.messages.append({"role": "user", "content": content})


def append_assistant_message(content: str, result: Dict[str, Any], mode: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "sources": result.get("sources", []),
            "meta": {
                "status": result.get("status", "-"),
                "best_score": result.get("best_score", 0.0),
                "mode": mode,
                "chat_message_id": result.get("chat_message_id"),
                "task_type": result.get("task_type"),
                "tool_used": result.get("tool_used"),
                "memory_used": result.get("memory_used", False),
                "memory_history_count": result.get("memory_history_count", 0),
            },
        }
    )


def submit_prompt(
    prompt: str,
    base_url: str,
    api_key: str,
    mode: str,
    category: str,
    vector_top_k: int,
    final_top_k: int,
    threshold: float,
    use_hybrid: bool,
    use_rerank: bool,
    use_memory: bool,
    memory_turns: int,
) -> None:
    append_user_message(prompt)
    with st.chat_message("user"):
        st.write(prompt)

    payload = {
        "session_id": st.session_state.session_id,
        "query": prompt,
        "use_memory": use_memory,
        "memory_turns": memory_turns,
        "category": category,
        "vector_top_k": vector_top_k,
        "final_top_k": final_top_k,
        "threshold": threshold,
        "use_hybrid": use_hybrid,
        "use_rerank": use_rerank,
        "save_log": True,
    }
    endpoint = "/agent/chat" if mode == "Agent RAG" else "/chat"

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and thinking..."):
            try:
                result = api_post(base_url, endpoint, payload, api_key)
                st.write(result.get("answer", ""))
                if result.get("memory_used"):
                    st.caption(f"Used {result.get('memory_history_count', 0)} previous turns from this session.")
                append_assistant_message(result.get("answer", ""), result, mode)
            except Exception as exc:
                message = f"Request failed: {exc}"
                st.error(message)
                append_assistant_message(
                    message,
                    {
                        "status": "failed",
                        "best_score": 0.0,
                        "sources": [],
                        "memory_used": False,
                        "memory_history_count": 0,
                    },
                    mode,
                )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="RAG", layout="wide")
    init_session_state()
    inject_style()

    st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Agent-ready research workspace for local paper RAG.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Workspace")
        base_url = st.text_input("API URL", value=DEFAULT_API_URL)
        api_key = st.text_input("API key", value=DEFAULT_API_KEY, type="password")

        col_a, col_b = st.columns(2)
        if col_a.button("Refresh", use_container_width=True):
            clear_api_caches()
            st.rerun()
        if col_b.button("New chat", use_container_width=True):
            reset_chat()
            st.rerun()

        st.caption(f"Session: {st.session_state.session_id[:12]}")
        st.caption("New chat starts a fresh session. Old messages stay in SQLite, but this page will not use them as memory.")

        st.header("Mode")
        mode = st.radio("Answer mode", ["Normal RAG", "Agent RAG"], horizontal=False)
        vector_top_k = st.slider("Vector top-k", min_value=3, max_value=20, value=8)
        final_top_k = st.slider("Final top-k", min_value=1, max_value=8, value=3)
        threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        use_hybrid = st.checkbox("Hybrid search", value=False)
        use_rerank = st.checkbox("Rerank", value=True)
        use_memory = st.checkbox("Use session memory", value=True)
        memory_turns = st.slider("Memory turns", min_value=1, max_value=10, value=4)

        st.header("Upload")
        uploaded_file = st.file_uploader("Document", type=["pdf", "txt", "md", "docx"])
        if st.button("Upload and index", disabled=uploaded_file is None, use_container_width=True):
            try:
                result = api_upload(base_url, "/documents/upload", uploaded_file, api_key)
                clear_api_caches()
                st.success(f"Queued: {result['filename']}")
            except Exception as exc:
                st.error(str(exc))

        with st.expander("Examples"):
            for example in EXAMPLE_QUERIES:
                if st.button(example, key=example, use_container_width=True):
                    st.session_state.pending_prompt = example
                    st.rerun()

    state = load_backend_state(base_url, api_key)
    sources_info = state.get("sources") or {}
    categories = sources_info.get("category_options", ["all"])
    category = "all"
    if categories:
        category = st.sidebar.selectbox("Category", categories, index=0)

    render_workspace_summary(state)

    left, right = st.columns([0.68, 0.32], gap="large")
    with left:
        with st.container(border=True):
            render_chat_history()

        prompt = st.chat_input("Ask about agent security, papers, methods, or comparisons")
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        if prompt:
            submit_prompt(
                prompt=prompt,
                base_url=base_url,
                api_key=api_key,
                mode=mode,
                category=category,
                vector_top_k=vector_top_k,
                final_top_k=final_top_k,
                threshold=threshold,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_memory=use_memory,
                memory_turns=memory_turns,
            )
            st.rerun()

    with right:
        with st.container(border=True):
            render_document_list(state.get("documents"))


if __name__ == "__main__":
    main()
