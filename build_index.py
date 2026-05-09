import os
import json
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    ensure_dirs,
    INDEX_DIR,
    LOG_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    INDEX_META_PATH,
    EMBEDDING_MODEL_NAME
)
from document_loader import load_documents
from text_splitter import build_chunks


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_index():
    ensure_dirs()

    print("正在读取文档...")
    documents = load_documents()
    print(f"读取到文档数量：{len(documents)}")

    if len(documents) == 0:
        raise ValueError("没有读取到任何文档，请检查 data 文件夹。")

    print("正在切分文档...")
    chunks = build_chunks(documents)
    print(f"生成 chunk 数量：{len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("没有生成任何 chunk，请检查文档内容或切分参数。")

    print("正在加载 embedding 模型...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("正在向量化 chunks...")
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings).astype("float32")

    if len(embeddings.shape) != 2:
        raise ValueError(f"embedding 维度异常：{embeddings.shape}")

    dimension = embeddings.shape[1]

    print("正在建立 FAISS 索引...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    save_json(chunks, CHUNKS_PATH)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_dimension": dimension,
        "faiss_index_type": "IndexFlatIP",
        "normalize_embeddings": True,
        "documents": [
            {
                "source": doc["source"],
                "extension": doc["extension"],
                "path": doc["path"],
                "text_length": len(doc["text"])
            }
            for doc in documents
        ]
    }

    save_json(meta, INDEX_META_PATH)

    os.makedirs(LOG_DIR, exist_ok=True)
    save_json(meta, os.path.join(LOG_DIR, "index_build_log.json"))

    print("索引构建完成！")
    print(f"FAISS 索引保存到：{FAISS_INDEX_PATH}")
    print(f"chunks 保存到：{CHUNKS_PATH}")
    print(f"索引元信息保存到：{INDEX_META_PATH}")


if __name__ == "__main__":
    build_index()