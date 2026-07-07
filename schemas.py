from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "rag-demo-api"


class SourceItem(BaseModel):
    source: str = ""
    doc_id: Optional[str] = None
    chunk_id: Optional[Any] = None
    matched_child_id: Optional[Any] = None
    chunk_uid: Optional[str] = None
    vector_id: Optional[int] = None
    paper_title: Optional[str] = None
    section_title: Optional[str] = None
    section_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    parent_id: Optional[str] = None
    context_mode: Optional[str] = None
    category: str = "general"
    relative_path: str = ""
    path: str = ""
    extension: str = ""
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    use_memory: bool = True
    memory_turns: int = Field(default=4, ge=1, le=10)
    category: str = "all"
    vector_top_k: Optional[int] = Field(default=None, ge=1, le=50)
    final_top_k: Optional[int] = Field(default=None, ge=1, le=20)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_hybrid: bool = False
    use_rerank: bool = True
    save_log: bool = True


class ChatResponse(BaseModel):
    chat_message_id: Optional[int] = None
    memory_used: bool = False
    memory_history_count: int = 0
    query: str
    answer: str
    status: str
    best_score: float
    use_hybrid: bool = False
    use_rerank: bool = False
    sources: List[SourceItem] = []


class AgentChatResponse(ChatResponse):
    task_type: str
    tool_used: str


class IndexStatusResponse(BaseModel):
    faiss_index_exists: bool
    chunks_exists: bool
    index_meta_exists: bool
    index_manifest_exists: bool
    chunk_count: int
    document_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    embedding_max_seq_length: Optional[int] = None
    faiss_index_type: Optional[str] = None
    chunk_strategy: Optional[str] = None
    chunk_token_size: Optional[int] = None
    chunk_token_overlap: Optional[int] = None
    parent_token_size: Optional[int] = None
    parent_token_overlap: Optional[int] = None
    manifest_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    documents: List[Dict[str, Any]] = []


class SourcesResponse(BaseModel):
    source_count: int
    sources: List[Dict[str, Any]]
    category_options: List[str]


class UploadResponse(BaseModel):
    index_job_id: Optional[str] = None
    document_id: str
    filename: str
    original_filename: str
    extension: str
    sha256: str
    size_bytes: int
    storage_path: str
    relative_path: str
    status: str
    indexed: bool = False
    created_at: Optional[str] = None
    queued_at: Optional[str] = None
    index_started_at: Optional[str] = None
    index_finished_at: Optional[str] = None
    error_message: Optional[str] = None
    index_result: Optional[Dict[str, Any]] = None


class DocumentsResponse(BaseModel):
    document_count: int
    documents: List[UploadResponse]
    db_path: Optional[str] = None
    manifest_path: Optional[str] = None
    updated_at: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
