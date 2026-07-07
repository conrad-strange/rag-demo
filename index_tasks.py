import threading
from typing import Dict, List, Optional

from index_manager import update_index
from storage import list_uploaded_documents, update_uploaded_document_status


_index_lock = threading.Lock()


def queued_documents() -> List[Dict]:
    return [
        document
        for document in list_uploaded_documents().get("documents", [])
        if document.get("status") == "queued"
    ]


def run_index_update_for_document(document_id: str, relative_path: str) -> Dict:
    if not _index_lock.acquire(blocking=False):
        update_uploaded_document_status(
            document_id,
            status="queued",
            indexed=False,
            error_message="Another index update is running; this document remains queued.",
        )
        return {
            "status": "queued",
            "message": "Another index update is running; document remains queued.",
        }

    try:
        processed_count = 0
        last_result = None

        while True:
            batch = queued_documents()
            if not batch:
                return {
                    "status": "idle",
                    "processed_count": processed_count,
                    "index_result": last_result,
                }

            for document in batch:
                update_uploaded_document_status(
                    document["document_id"],
                    status="indexing",
                    indexed=False,
                    error_message=None,
                )

            try:
                result = update_index()
                clear_rag_service_caches()
            except Exception as exc:
                for document in batch:
                    update_uploaded_document_status(
                        document["document_id"],
                        status="failed",
                        indexed=False,
                        error_message=str(exc),
                    )
                return {
                    "status": "failed",
                    "relative_path": relative_path,
                    "processed_count": processed_count,
                    "error_message": str(exc),
                }

            for document in batch:
                update_uploaded_document_status(
                    document["document_id"],
                    status="indexed",
                    indexed=True,
                    error_message=None,
                    index_result=result,
                )
            processed_count += len(batch)
            last_result = result
    except Exception as exc:
        update_uploaded_document_status(
            document_id,
            status="failed",
            indexed=False,
            error_message=str(exc),
        )
        return {
            "status": "failed",
            "relative_path": relative_path,
            "error_message": str(exc),
        }
    finally:
        _index_lock.release()


def trigger_index_update_for_document(document: Dict) -> Dict:
    return run_index_update_for_document(
        document_id=document["document_id"],
        relative_path=document["relative_path"],
    )


def clear_rag_service_caches() -> None:
    try:
        import rag_service

        cache_functions = [
            "_cached_index_meta",
            "_cached_chunks",
            "get_rag_pipeline",
            "get_cached_agent_workflow",
        ]
        for name in cache_functions:
            func: Optional[object] = getattr(rag_service, name, None)
            if func is not None and hasattr(func, "cache_clear"):
                func.cache_clear()
    except Exception:
        pass
