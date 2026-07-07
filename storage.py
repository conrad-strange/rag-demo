import hashlib
import json
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
    UPLOAD_MANIFEST_PATH,
)


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
    with _manifest_lock:
        ensure_upload_dir()
        if not os.path.exists(UPLOAD_MANIFEST_PATH):
            return {"schema_version": 1, "documents": []}
        with open(UPLOAD_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("schema_version", 1)
    data.setdefault("documents", [])
    return data


def save_upload_manifest(manifest: Dict) -> None:
    with _manifest_lock:
        ensure_upload_dir()
        tmp_path = f"{UPLOAD_MANIFEST_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, UPLOAD_MANIFEST_PATH)


def record_uploaded_document(record: Dict) -> None:
    with _manifest_lock:
        manifest = load_upload_manifest()
        documents = manifest.setdefault("documents", [])
        documents.append(record)
        manifest["updated_at"] = now_str()
        save_upload_manifest(manifest)


def update_uploaded_document_status(
    document_id: str,
    status: str,
    indexed: Optional[bool] = None,
    error_message: Optional[str] = None,
    index_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    with _manifest_lock:
        manifest = load_upload_manifest()
        now = now_str()
        for document in manifest.get("documents", []):
            if document.get("document_id") != document_id:
                continue

            document["status"] = status
            if indexed is not None:
                document["indexed"] = indexed
            if error_message is None:
                document.pop("error_message", None)
            else:
                document["error_message"] = error_message
            if index_result is not None:
                document["index_result"] = index_result

            if status == "queued":
                document["queued_at"] = now
            elif status == "indexing":
                document["index_started_at"] = now
            elif status in ("indexed", "failed"):
                document["index_finished_at"] = now

            manifest["updated_at"] = now
            save_upload_manifest(manifest)
            return document
    return None


def list_uploaded_documents() -> Dict:
    manifest = load_upload_manifest()
    documents: List[Dict] = list(manifest.get("documents", []))
    documents.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {
        "document_count": len(documents),
        "documents": documents,
        "manifest_path": UPLOAD_MANIFEST_PATH,
        "updated_at": manifest.get("updated_at"),
    }


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
