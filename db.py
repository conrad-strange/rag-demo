import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from config import DATABASE_PATH, UPLOAD_MANIFEST_PATH


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                extension TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                storage_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                indexed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                queued_at TEXT,
                index_started_at TEXT,
                index_finished_at TEXT,
                error_message TEXT,
                index_result_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_jobs (
                job_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                index_result_json TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                mode TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                status TEXT NOT NULL,
                best_score REAL,
                use_hybrid INTEGER NOT NULL DEFAULT 0,
                use_rerank INTEGER NOT NULL DEFAULT 0,
                sources_json TEXT,
                task_type TEXT,
                tool_used TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def row_to_document(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    item["indexed"] = bool(item.get("indexed"))
    if item.get("index_result_json"):
        item["index_result"] = json.loads(item.pop("index_result_json"))
    else:
        item.pop("index_result_json", None)
        item["index_result"] = None
    return item


def insert_document(record: Dict[str, Any]) -> Dict[str, Any]:
    init_db()
    data = record.copy()
    for key in (
        "queued_at",
        "index_started_at",
        "index_finished_at",
        "error_message",
        "index_result",
    ):
        data.setdefault(key, None)
    data.setdefault("updated_at", data.get("created_at") or now_str())
    data["indexed"] = 1 if data.get("indexed") else 0
    data["index_result_json"] = json.dumps(data.get("index_result"), ensure_ascii=False) if data.get("index_result") is not None else None
    with connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents (
                document_id, filename, original_filename, extension, sha256,
                size_bytes, storage_path, relative_path, status, indexed,
                created_at, updated_at, queued_at, index_started_at,
                index_finished_at, error_message, index_result_json
            )
            VALUES (
                :document_id, :filename, :original_filename, :extension, :sha256,
                :size_bytes, :storage_path, :relative_path, :status, :indexed,
                :created_at, :updated_at, :queued_at, :index_started_at,
                :index_finished_at, :error_message, :index_result_json
            )
            """,
            data,
        )
    if record.get("status") == "queued":
        job = create_index_job(record["document_id"], status="queued")
        record["index_job_id"] = job["job_id"]
    return record


def create_index_job(document_id: str, status: str = "queued") -> Dict[str, Any]:
    init_db()
    job = {
        "job_id": uuid.uuid4().hex,
        "document_id": document_id,
        "status": status,
        "created_at": now_str(),
        "started_at": None,
        "finished_at": None,
        "error_message": None,
        "index_result_json": None,
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO index_jobs (
                job_id, document_id, status, created_at, started_at,
                finished_at, error_message, index_result_json
            )
            VALUES (
                :job_id, :document_id, :status, :created_at, :started_at,
                :finished_at, :error_message, :index_result_json
            )
            """,
            job,
        )
    return job


def latest_index_job(document_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM index_jobs
            WHERE document_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def update_document_status(
    document_id: str,
    status: str,
    indexed: Optional[bool] = None,
    error_message: Optional[str] = None,
    index_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    init_db()
    now = now_str()
    fields = {
        "status": status,
        "updated_at": now,
        "error_message": error_message,
    }
    if indexed is not None:
        fields["indexed"] = 1 if indexed else 0
    if index_result is not None:
        fields["index_result_json"] = json.dumps(index_result, ensure_ascii=False)
    if status == "queued":
        fields["queued_at"] = now
    elif status == "indexing":
        fields["index_started_at"] = now
    elif status in ("indexed", "failed"):
        fields["index_finished_at"] = now

    assignments = ", ".join(f"{key} = :{key}" for key in fields)
    fields["document_id"] = document_id
    with connect() as conn:
        conn.execute(
            f"UPDATE documents SET {assignments} WHERE document_id = :document_id",
            fields,
        )

    update_latest_index_job(document_id, status, error_message=error_message, index_result=index_result)
    return get_document(document_id)


def update_latest_index_job(
    document_id: str,
    status: str,
    error_message: Optional[str] = None,
    index_result: Optional[Dict[str, Any]] = None,
) -> None:
    job = latest_index_job(document_id)
    if job is None:
        job = create_index_job(document_id, status=status)
    now = now_str()
    fields = {
        "status": status,
        "error_message": error_message,
        "index_result_json": json.dumps(index_result, ensure_ascii=False) if index_result is not None else None,
        "job_id": job["job_id"],
    }
    if status == "indexing":
        fields["started_at"] = now
    elif status in ("indexed", "failed"):
        fields["finished_at"] = now

    assignments = ", ".join(f"{key} = :{key}" for key in fields if key != "job_id")
    with connect() as conn:
        conn.execute(
            f"UPDATE index_jobs SET {assignments} WHERE job_id = :job_id",
            fields,
        )


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return row_to_document(row) if row else None


def list_documents() -> Dict[str, Any]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    documents = [row_to_document(row) for row in rows]
    return {
        "document_count": len(documents),
        "documents": documents,
        "db_path": DATABASE_PATH,
        "updated_at": documents[0]["updated_at"] if documents else None,
    }


def list_documents_by_status(statuses: Iterable[str]) -> List[Dict[str, Any]]:
    init_db()
    placeholders = ", ".join("?" for _ in statuses)
    values = list(statuses)
    if not values:
        return []
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY created_at ASC",
            values,
        ).fetchall()
    return [row_to_document(row) for row in rows]


def record_chat_message(
    mode: str,
    query: str,
    answer: str,
    status: str,
    best_score: Optional[float] = None,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    sources: Optional[List[Dict[str, Any]]] = None,
    task_type: Optional[str] = None,
    tool_used: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    init_db()
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (
                session_id, mode, query, answer, status, best_score,
                use_hybrid, use_rerank, sources_json, task_type,
                tool_used, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                mode,
                query,
                answer,
                status,
                best_score,
                1 if use_hybrid else 0,
                1 if use_rerank else 0,
                json.dumps(sources or [], ensure_ascii=False),
                task_type,
                tool_used,
                now_str(),
            ),
        )
        return int(cursor.lastrowid)


def list_recent_chat_messages(session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    init_db()
    if not session_id:
        return []
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, int(limit)),
        ).fetchall()

    messages = [dict(row) for row in rows]
    messages.reverse()
    for message in messages:
        if message.get("sources_json"):
            message["sources"] = json.loads(message.pop("sources_json"))
        else:
            message.pop("sources_json", None)
            message["sources"] = []
        message["use_hybrid"] = bool(message.get("use_hybrid"))
        message["use_rerank"] = bool(message.get("use_rerank"))
    return messages


def migrate_upload_manifest(manifest_path: str = UPLOAD_MANIFEST_PATH) -> None:
    if not os.path.exists(manifest_path):
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    migrated = False
    for document in data.get("documents", []):
        if not document.get("document_id"):
            continue
        if get_document(document["document_id"]) is not None:
            continue
        insert_document(document)
        migrated = True

    if migrated:
        data["migrated_to_sqlite_at"] = now_str()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
