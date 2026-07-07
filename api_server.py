from typing import Callable

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status

from auth import require_api_key
from rag_service import (
    agent_security_answer,
    answer_security_question,
    get_index_status,
    list_knowledge_sources,
)
from schemas import (
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    DocumentsResponse,
    ErrorResponse,
    HealthResponse,
    IndexStatusResponse,
    SourcesResponse,
    UploadResponse,
)
from storage import UploadValidationError, list_uploaded_documents, save_upload_file
from index_tasks import trigger_index_update_for_document


app = FastAPI(
    title="AI Research Paper RAG API",
    version="0.1.0",
    description="HTTP API wrapper around the local RAG and Agent RAG service layer.",
)


def _service_call(func: Callable, **kwargs):
    try:
        return func(**kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG service error: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get(
    "/index/status",
    response_model=IndexStatusResponse,
    responses={500: {"model": ErrorResponse}},
)
def index_status() -> dict:
    return _service_call(get_index_status)


@app.get(
    "/sources",
    response_model=SourcesResponse,
    responses={500: {"model": ErrorResponse}},
)
def sources() -> dict:
    return _service_call(list_knowledge_sources)


@app.get(
    "/documents",
    response_model=DocumentsResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def documents(_: None = Depends(require_api_key)) -> dict:
    return list_uploaded_documents()


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
) -> dict:
    try:
        document = save_upload_file(file.file, file.filename or "", status="queued")
        background_tasks.add_task(trigger_index_update_for_document, document)
        return document
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        file.file.close()


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def chat(
    request: ChatRequest,
    _: None = Depends(require_api_key),
) -> dict:
    return _service_call(
        answer_security_question,
        query=request.query,
        category=request.category,
        vector_top_k=request.vector_top_k,
        final_top_k=request.final_top_k,
        threshold=request.threshold,
        use_hybrid=request.use_hybrid,
        use_rerank=request.use_rerank,
        save_log=request.save_log,
    )


@app.post(
    "/agent/chat",
    response_model=AgentChatResponse,
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def agent_chat(
    request: ChatRequest,
    _: None = Depends(require_api_key),
) -> dict:
    return _service_call(
        agent_security_answer,
        query=request.query,
        category=request.category,
        vector_top_k=request.vector_top_k,
        final_top_k=request.final_top_k,
        threshold=request.threshold,
        use_hybrid=request.use_hybrid,
        use_rerank=request.use_rerank,
        save_log=request.save_log,
    )
