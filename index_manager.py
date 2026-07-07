import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STRATEGY_VERSION,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    CHUNKS_PATH,
    DATA_DIR,
    DOCUMENT_CATEGORIES,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_MANIFEST_PATH,
    INDEX_META_PATH,
    LOG_DIR,
    MIN_CHUNK_SIZE,
    MIN_CHUNK_TOKENS,
    PARENT_TOKEN_OVERLAP,
    PARENT_TOKEN_SIZE,
    SUPPORTED_EXTENSIONS,
    ensure_dirs,
)
from document_loader import read_document
from embedding_utils import resolve_embedding_device
from text_splitter import build_chunks


MANIFEST_SCHEMA_VERSION = 1
VECTOR_ID_START = 1


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(data, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def load_json(file_path: str, default):
    if not os.path.exists(file_path):
        return default
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_doc_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def data_relative_path(file_path: str) -> str:
    return Path(file_path).resolve().relative_to(Path(DATA_DIR).resolve()).as_posix()


def iter_source_files(data_dir: str = DATA_DIR) -> Iterable[str]:
    data_path = Path(data_dir)
    extensions = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    for path in sorted(data_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield str(path)


def file_record(file_path: str) -> Dict:
    abs_path = str(Path(file_path).resolve())
    relative_path = data_relative_path(abs_path)
    source = os.path.basename(abs_path)
    stat = os.stat(abs_path)
    return {
        "doc_id": stable_doc_id(relative_path),
        "source": source,
        "relative_path": relative_path,
        "path": abs_path,
        "extension": os.path.splitext(source)[1].lower(),
        "category": DOCUMENT_CATEGORIES.get(source, "general"),
        "file_hash": file_sha256(abs_path),
        "file_size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def current_file_records() -> Dict[str, Dict]:
    return {
        record["relative_path"]: record
        for record in (file_record(path) for path in iter_source_files())
    }


def manifest_settings() -> Dict:
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "embedding_max_seq_length": EMBEDDING_MAX_SEQ_LENGTH,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_chunk_size": MIN_CHUNK_SIZE,
        "chunk_strategy": CHUNK_STRATEGY_VERSION,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "chunk_token_overlap": CHUNK_TOKEN_OVERLAP,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "parent_token_size": PARENT_TOKEN_SIZE,
        "parent_token_overlap": PARENT_TOKEN_OVERLAP,
        "normalize_embeddings": True,
        "faiss_index_type": "IndexIDMap2(IndexFlatIP)",
    }


def default_manifest() -> Dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": now_str(),
        "updated_at": now_str(),
        "next_vector_id": VECTOR_ID_START,
        "embedding_dimension": None,
        "documents": {},
        **manifest_settings(),
    }


def load_manifest() -> Dict:
    manifest = load_json(INDEX_MANIFEST_PATH, default=None)
    if manifest is None:
        return default_manifest()
    manifest.setdefault("documents", {})
    manifest.setdefault("next_vector_id", VECTOR_ID_START)
    return manifest


def manifest_config_mismatches(manifest: Dict) -> List[str]:
    mismatches = []
    for key, expected in manifest_settings().items():
        if manifest.get(key) != expected:
            mismatches.append(f"{key}: manifest={manifest.get(key)!r}, current={expected!r}")
    return mismatches


def scan_index() -> Dict:
    manifest = load_manifest()
    current = current_file_records()
    indexed = manifest.get("documents", {})

    added = []
    modified = []
    unchanged = []
    deleted = []

    for relative_path, record in current.items():
        old = indexed.get(relative_path)
        if old is None:
            added.append(record)
        elif old.get("file_hash") != record["file_hash"]:
            modified.append({**record, "previous_file_hash": old.get("file_hash")})
        else:
            unchanged.append(record)

    for relative_path, old in indexed.items():
        if relative_path not in current:
            deleted.append(old)

    mismatches = manifest_config_mismatches(manifest) if os.path.exists(INDEX_MANIFEST_PATH) else []
    return {
        "index_manifest_exists": os.path.exists(INDEX_MANIFEST_PATH),
        "faiss_index_exists": os.path.exists(FAISS_INDEX_PATH),
        "chunks_exists": os.path.exists(CHUNKS_PATH),
        "config_mismatches": mismatches,
        "counts": {
            "added": len(added),
            "modified": len(modified),
            "unchanged": len(unchanged),
            "deleted": len(deleted),
        },
        "added": added,
        "modified": modified,
        "unchanged": unchanged,
        "deleted": deleted,
    }


def document_from_record(record: Dict) -> Optional[Dict]:
    text = read_document(record["path"])
    if not text or not text.strip():
        print(f"[skip] Empty or unreadable document: {record['path']}")
        return None
    return {
        "doc_id": record["doc_id"],
        "source": record["source"],
        "relative_path": record["relative_path"],
        "path": record["path"],
        "extension": record["extension"],
        "category": record["category"],
        "file_hash": record["file_hash"],
        "text": text,
    }


def chunks_for_document(record: Dict) -> List[Dict]:
    document = document_from_record(record)
    if document is None:
        return []

    chunks = build_chunks([document])
    indexed_at = now_str()
    for chunk in chunks:
        chunk["doc_id"] = record["doc_id"]
        chunk["relative_path"] = record["relative_path"]
        chunk["file_hash"] = record["file_hash"]
        chunk["chunk_uid"] = f"{record['doc_id']}:{chunk['chunk_id']}"
        chunk["indexed_at"] = indexed_at
    return chunks


def assign_vector_ids(chunks: List[Dict], manifest: Dict) -> None:
    next_vector_id = int(manifest.get("next_vector_id", VECTOR_ID_START))
    for chunk in chunks:
        chunk["vector_id"] = next_vector_id
        next_vector_id += 1
    manifest["next_vector_id"] = next_vector_id


def load_embedder():
    from sentence_transformers import SentenceTransformer

    device = resolve_embedding_device(EMBEDDING_DEVICE)
    print(f"Loading embedding model on device: {device}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    if EMBEDDING_MAX_SEQ_LENGTH:
        embedder.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        try:
            embedder[0].max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        except Exception:
            pass
    return embedder


def embed_chunks(chunks: List[Dict]):
    if not chunks:
        return np.empty((0, 0), dtype="float32")

    embedder = load_embedder()
    embeddings = embedder.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    return np.asarray(embeddings, dtype="float32")


def create_index(dimension: int):
    import faiss

    base_index = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap2(base_index)


def read_faiss_index():
    import faiss

    return faiss.read_index(FAISS_INDEX_PATH)


def is_index_id_map(index) -> bool:
    return type(index).__name__ == "IndexIDMap2"


def add_chunks_to_index(index, chunks: List[Dict], embeddings) -> None:
    if not chunks:
        return
    ids = np.asarray([chunk["vector_id"] for chunk in chunks], dtype="int64")
    index.add_with_ids(embeddings, ids)


def remove_vector_ids(index, vector_ids: List[int]) -> int:
    if not vector_ids:
        return 0
    ids = np.asarray(vector_ids, dtype="int64")
    return int(index.remove_ids(ids))


def document_summary(record: Dict, chunks: List[Dict]) -> Dict:
    vector_ids = [int(chunk["vector_id"]) for chunk in chunks]
    parent_ids = {chunk.get("parent_id") for chunk in chunks if chunk.get("parent_id")}
    section_keys = {
        (chunk.get("section_index"), chunk.get("section_title"))
        for chunk in chunks
        if chunk.get("section_title")
    }
    return {
        "doc_id": record["doc_id"],
        "source": record["source"],
        "relative_path": record["relative_path"],
        "path": record["path"],
        "extension": record["extension"],
        "category": record["category"],
        "file_hash": record["file_hash"],
        "file_size": record["file_size"],
        "modified_at": record["modified_at"],
        "indexed_at": now_str(),
        "chunk_count": len(chunks),
        "parent_count": len(parent_ids),
        "section_count": len(section_keys),
        "vector_ids": vector_ids,
        "min_vector_id": min(vector_ids) if vector_ids else None,
        "max_vector_id": max(vector_ids) if vector_ids else None,
    }


def make_index_meta(manifest: Dict, chunks: List[Dict]) -> Dict:
    documents = sorted(
        manifest.get("documents", {}).values(),
        key=lambda item: item.get("relative_path", ""),
    )
    return {
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "embedding_model": manifest.get("embedding_model"),
        "embedding_batch_size": manifest.get("embedding_batch_size"),
        "embedding_max_seq_length": manifest.get("embedding_max_seq_length"),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_dimension": manifest.get("embedding_dimension"),
        "faiss_index_type": manifest.get("faiss_index_type"),
        "normalize_embeddings": manifest.get("normalize_embeddings", True),
        "chunk_strategy": manifest.get("chunk_strategy"),
        "chunk_token_size": manifest.get("chunk_token_size"),
        "chunk_token_overlap": manifest.get("chunk_token_overlap"),
        "parent_token_size": manifest.get("parent_token_size"),
        "parent_token_overlap": manifest.get("parent_token_overlap"),
        "manifest_path": INDEX_MANIFEST_PATH,
        "documents": [
            {
                "doc_id": doc.get("doc_id"),
                "source": doc.get("source"),
                "relative_path": doc.get("relative_path"),
                "extension": doc.get("extension"),
                "path": doc.get("path"),
                "category": doc.get("category", "general"),
                "file_hash": doc.get("file_hash"),
                "chunk_count": doc.get("chunk_count", 0),
                "parent_count": doc.get("parent_count", 0),
                "section_count": doc.get("section_count", 0),
                "indexed_at": doc.get("indexed_at"),
            }
            for doc in documents
        ],
    }


def save_index_outputs(index, chunks: List[Dict], manifest: Dict) -> Dict:
    import faiss

    manifest["updated_at"] = now_str()
    meta = make_index_meta(manifest, chunks)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_json(chunks, CHUNKS_PATH)
    save_json(manifest, INDEX_MANIFEST_PATH)
    save_json(meta, INDEX_META_PATH)

    os.makedirs(LOG_DIR, exist_ok=True)
    save_json(meta, os.path.join(LOG_DIR, "index_build_log.json"))
    return meta


def rebuild_index(reason: str = "manual rebuild") -> Dict:
    ensure_dirs()
    records = list(current_file_records().values())
    if not records:
        raise ValueError("No supported documents found under data/.")

    print(f"Rebuilding index: {reason}")
    print(f"Documents found: {len(records)}")

    manifest = default_manifest()
    all_chunks: List[Dict] = []

    for record in records:
        chunks = chunks_for_document(record)
        assign_vector_ids(chunks, manifest)
        all_chunks.extend(chunks)
        manifest["documents"][record["relative_path"]] = document_summary(record, chunks)
        print(f"- {record['relative_path']}: {len(chunks)} chunks")

    if not all_chunks:
        raise ValueError("No chunks generated from documents.")

    print(f"Embedding chunks: {len(all_chunks)}")
    embeddings = embed_chunks(all_chunks)
    if len(embeddings.shape) != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    dimension = int(embeddings.shape[1])
    manifest["embedding_dimension"] = dimension

    print("Building FAISS IndexIDMap2(IndexFlatIP)...")
    index = create_index(dimension)
    add_chunks_to_index(index, all_chunks, embeddings)

    meta = save_index_outputs(index, all_chunks, manifest)
    print("Index rebuild complete.")
    return {
        "action": "rebuild",
        "reason": reason,
        "document_count": meta["document_count"],
        "chunk_count": meta["chunk_count"],
        "embedding_dimension": dimension,
        "faiss_index_type": meta["faiss_index_type"],
    }


def update_index() -> Dict:
    ensure_dirs()

    if not (
        os.path.exists(INDEX_MANIFEST_PATH)
        and os.path.exists(FAISS_INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
    ):
        return rebuild_index(reason="missing index, chunks, or manifest")

    manifest = load_manifest()
    mismatches = manifest_config_mismatches(manifest)
    if mismatches:
        raise RuntimeError(
            "Index settings changed. Run `python index_manager.py rebuild`.\n"
            + "\n".join(mismatches)
        )

    index = read_faiss_index()
    if not is_index_id_map(index):
        return rebuild_index(reason="existing FAISS index is not IndexIDMap2")

    chunks = load_json(CHUNKS_PATH, default=[])
    scan = scan_index()
    changed = scan["added"] + scan["modified"]
    removed_records = scan["deleted"] + [
        manifest["documents"][item["relative_path"]]
        for item in scan["modified"]
        if item["relative_path"] in manifest.get("documents", {})
    ]

    if not changed and not scan["deleted"]:
        return {
            "action": "update",
            "status": "unchanged",
            "document_count": len(manifest.get("documents", {})),
            "chunk_count": len(chunks),
            "faiss_ntotal": int(index.ntotal),
        }

    remove_paths = {record["relative_path"] for record in removed_records}
    remove_ids = []
    for record in removed_records:
        remove_ids.extend(int(vector_id) for vector_id in record.get("vector_ids", []))

    removed_vector_count = remove_vector_ids(index, remove_ids)
    chunks = [
        chunk for chunk in chunks
        if chunk.get("relative_path") not in remove_paths
    ]
    for relative_path in remove_paths:
        manifest["documents"].pop(relative_path, None)

    new_chunks: List[Dict] = []
    for record in changed:
        chunks_for_doc = chunks_for_document(record)
        assign_vector_ids(chunks_for_doc, manifest)
        new_chunks.extend(chunks_for_doc)
        manifest["documents"][record["relative_path"]] = document_summary(record, chunks_for_doc)
        print(f"- indexed {record['relative_path']}: {len(chunks_for_doc)} chunks")

    if new_chunks:
        embeddings = embed_chunks(new_chunks)
        if manifest.get("embedding_dimension") is None:
            manifest["embedding_dimension"] = int(embeddings.shape[1])
        add_chunks_to_index(index, new_chunks, embeddings)
        chunks.extend(new_chunks)

    meta = save_index_outputs(index, chunks, manifest)
    return {
        "action": "update",
        "added": len(scan["added"]),
        "modified": len(scan["modified"]),
        "deleted": len(scan["deleted"]),
        "removed_vector_count": removed_vector_count,
        "new_chunk_count": len(new_chunks),
        "document_count": meta["document_count"],
        "chunk_count": meta["chunk_count"],
        "faiss_ntotal": int(index.ntotal),
    }


def find_manifest_document(manifest: Dict, source: str) -> Dict:
    docs = list(manifest.get("documents", {}).values())
    matches = [
        doc for doc in docs
        if source in {
            doc.get("relative_path"),
            doc.get("source"),
            doc.get("doc_id"),
        }
    ]
    if not matches:
        raise ValueError(f"No indexed document found for: {source}")
    if len(matches) > 1:
        options = ", ".join(doc.get("relative_path", "") for doc in matches)
        raise ValueError(f"Ambiguous source {source!r}. Use relative_path. Matches: {options}")
    return matches[0]


def remove_document(source: str) -> Dict:
    ensure_dirs()
    manifest = load_manifest()
    index = read_faiss_index()
    if not is_index_id_map(index):
        raise RuntimeError("Current FAISS index is not IndexIDMap2. Run rebuild first.")

    target = find_manifest_document(manifest, source)
    vector_ids = [int(vector_id) for vector_id in target.get("vector_ids", [])]
    removed_vector_count = remove_vector_ids(index, vector_ids)

    chunks = load_json(CHUNKS_PATH, default=[])
    chunks = [
        chunk for chunk in chunks
        if chunk.get("relative_path") != target.get("relative_path")
    ]
    manifest["documents"].pop(target["relative_path"], None)

    meta = save_index_outputs(index, chunks, manifest)
    return {
        "action": "remove",
        "source": source,
        "relative_path": target.get("relative_path"),
        "removed_vector_count": removed_vector_count,
        "document_count": meta["document_count"],
        "chunk_count": meta["chunk_count"],
        "faiss_ntotal": int(index.ntotal),
        "note": "The source file was not deleted from data/. Running update will add it again if it still exists.",
    }


def index_status() -> Dict:
    manifest = load_manifest()
    chunks = load_json(CHUNKS_PATH, default=[])
    faiss_ntotal = None
    faiss_index_type = None

    if os.path.exists(FAISS_INDEX_PATH):
        index = read_faiss_index()
        faiss_ntotal = int(index.ntotal)
        faiss_index_type = type(index).__name__

    documents = []
    for doc in sorted(
        manifest.get("documents", {}).values(),
        key=lambda item: item.get("relative_path", ""),
    ):
        documents.append({
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "relative_path": doc.get("relative_path"),
            "category": doc.get("category"),
            "file_hash": doc.get("file_hash"),
            "chunk_count": doc.get("chunk_count", 0),
            "parent_count": doc.get("parent_count", 0),
            "section_count": doc.get("section_count", 0),
            "min_vector_id": doc.get("min_vector_id"),
            "max_vector_id": doc.get("max_vector_id"),
            "indexed_at": doc.get("indexed_at"),
        })

    return {
        "index_manifest_exists": os.path.exists(INDEX_MANIFEST_PATH),
        "faiss_index_exists": os.path.exists(FAISS_INDEX_PATH),
        "chunks_exists": os.path.exists(CHUNKS_PATH),
        "embedding_model": manifest.get("embedding_model"),
        "embedding_dimension": manifest.get("embedding_dimension"),
        "embedding_batch_size": manifest.get("embedding_batch_size"),
        "embedding_max_seq_length": manifest.get("embedding_max_seq_length"),
        "configured_faiss_index_type": manifest.get("faiss_index_type"),
        "actual_faiss_index_type": faiss_index_type,
        "chunk_strategy": manifest.get("chunk_strategy"),
        "chunk_token_size": manifest.get("chunk_token_size"),
        "chunk_token_overlap": manifest.get("chunk_token_overlap"),
        "parent_token_size": manifest.get("parent_token_size"),
        "parent_token_overlap": manifest.get("parent_token_overlap"),
        "document_count": len(manifest.get("documents", {})),
        "chunk_count": len(chunks),
        "faiss_ntotal": faiss_ntotal,
        "next_vector_id": manifest.get("next_vector_id"),
        "updated_at": manifest.get("updated_at"),
        "documents": documents,
    }


def print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local Security RAG FAISS index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("scan", help="Show added, modified, deleted, and unchanged documents.")
    subparsers.add_parser("status", help="Show current index and manifest status.")
    subparsers.add_parser("update", help="Incrementally update the index from data/.")
    subparsers.add_parser("rebuild", help="Fully rebuild the index from data/.")

    remove_parser = subparsers.add_parser("remove", help="Remove one indexed document from the vector store.")
    remove_parser.add_argument("--source", required=True, help="Document source, relative path, or doc_id.")

    args = parser.parse_args()

    if args.command == "scan":
        print_json(scan_index())
    elif args.command == "status":
        print_json(index_status())
    elif args.command == "update":
        print_json(update_index())
    elif args.command == "rebuild":
        print_json(rebuild_index())
    elif args.command == "remove":
        print_json(remove_document(args.source))


if __name__ == "__main__":
    main()
