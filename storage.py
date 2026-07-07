import hashlib
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Dict, List, Optional

from config import (
    DATA_DIR,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from db import insert_document, list_documents, migrate_upload_manifest, update_document_status


class UploadValidationError(ValueError):
    pass


_manifest_lock = RLock()


def sanitize_filename(filename: str) -> str:
    name = os.path.basename(filename or "").strip()
    if not name:
        raise UploadValidationError("Filename is required.")

    stem, ext = os.path.splitext(name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    safe_ext = ext.lower()
    if not safe_stem:
        safe_stem = "document"
    return f"{safe_stem}{safe_ext}"


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    allowed = {item.lower() for item in SUPPORTED_EXTENSIONS}
    if ext not in allowed:
        raise UploadValidationError(
            f"Unsupported file type {ext!r}. Allowed types: {', '.join(sorted(allowed))}."
        )
    return ext


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_upload_manifest() -> Dict:
    return list_documents()


def save_upload_manifest(manifest: Dict) -> None:
    raise RuntimeError("upload_manifest.json has been replaced by SQLite storage.")


def record_uploaded_document(record: Dict) -> None:
    with _manifest_lock:
        insert_document(record)


def update_uploaded_document_status(
    document_id: str,
    status: str,
    indexed: Optional[bool] = None,
    error_message: Optional[str] = None,
    index_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    with _manifest_lock:
        return update_document_status(
            document_id=document_id,
            status=status,
            indexed=indexed,
            error_message=error_message,
            index_result=index_result,
        )


def list_uploaded_documents() -> Dict:
    migrate_upload_manifest()
    return list_documents()


def save_upload_file(file_obj: BinaryIO, filename: str, status: str = "stored") -> Dict:
    safe_name = sanitize_filename(filename)
    extension = validate_extension(safe_name)
    document_id = uuid.uuid4().hex
    document_dir = Path(UPLOAD_DIR) / document_id
    document_dir.mkdir(parents=True, exist_ok=False)

    target_path = document_dir / safe_name
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        with open(target_path, "wb") as output:
            while True:
                block = file_obj.read(1024 * 1024)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise UploadValidationError(
                        f"File is too large. Max upload size is {MAX_UPLOAD_BYTES} bytes."
                    )
                digest.update(block)
                output.write(block)
    except Exception:
        if target_path.exists():
            target_path.unlink()
        try:
            document_dir.rmdir()
        except OSError:
            pass
        raise

    relative_path = target_path.resolve().relative_to(Path(DATA_DIR).resolve()).as_posix()
    record = {
        "document_id": document_id,
        "filename": safe_name,
        "original_filename": filename,
        "extension": extension,
        "sha256": digest.hexdigest(),
        "size_bytes": total_bytes,
        "storage_path": str(target_path.resolve()),
        "relative_path": relative_path,
        "status": status,
        "indexed": status == "indexed",
        "created_at": now_str(),
    }
    if status == "queued":
        record["queued_at"] = record["created_at"]
    record_uploaded_document(record)
    return record
