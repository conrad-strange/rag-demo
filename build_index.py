import os
import re
import glob
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
INDEX_DIR = "index"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_long_paragraph(paragraph, chunk_size=400, overlap=80):
    chunks = []
    start = 0

    while start < len(paragraph):
        end = start + chunk_size
        chunk = paragraph[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(paragraph):
            break

        start += chunk_size - overlap

    return chunks


def split_text_by_paragraph(text, chunk_size=400, overlap=80, min_chunk_size=80):
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            chunks.extend(
                split_long_paragraph(
                    para,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
            )
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())

            current_chunk = para

    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return chunks


def load_txt_docs(data_dir=DATA_DIR):
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    docs = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append({
            "source": os.path.basename(file_path),
            "text": text
        })

    return docs


def build_chunks(docs):
    all_chunks = []

    for doc in docs:
        chunks = split_text_by_paragraph(
            doc["text"],
            chunk_size=400,
            overlap=80,
            min_chunk_size=80
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
                "chunk_length": len(chunk)
            })

    return all_chunks


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("正在读取文档...")
    docs = load_txt_docs(DATA_DIR)
    print(f"读取到 {len(docs)} 个文档")

    print("正在切分文档...")
    all_chunks = build_chunks(docs)
    print(f"生成 {len(all_chunks)} 个 chunk")

    print("正在加载 embedding 模型...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [item["text"] for item in all_chunks]

    print("正在向量化...")
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings).astype("float32")

    print("正在建立 FAISS 索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("索引构建完成！")
    print("保存位置：index/faiss.index 和 index/chunks.json")


if __name__ == "__main__":
    main()