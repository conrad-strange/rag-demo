import re
from typing import List, Dict
from config import DOCUMENT_CATEGORIES
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


def clean_text(text: str) -> str:
    """
    基础清洗：
    1. 统一换行符；
    2. 合并过多空行；
    3. 删除多余空格；
    4. 保留段落结构。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_long_paragraph(
    paragraph: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    长段落使用滑动窗口切分。
    overlap 的作用是避免知识点刚好被切断。
    """
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


def split_text_by_paragraph(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE
) -> List[str]:
    """
    段落优先切分。
    这个方法比固定长度切分更适合中文知识文档。
    """
    text = clean_text(text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果段落本身过长，先处理当前缓存，再切分长段落
        if len(para) > chunk_size:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            chunks.extend(
                split_long_paragraph(
                    paragraph=para,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
            )
            continue

        # 如果当前 chunk 加上新段落不超长，则合并
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


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """
    把所有文档切成 chunks。
    """
    all_chunks = []
    
    for doc in documents:
        chunks = split_text_by_paragraph(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "path": doc["path"],
                "extension": doc["extension"],
                "category": DOCUMENT_CATEGORIES.get(doc["source"], "general"),
                "chunk_id": i,
                "chunk_length": len(chunk),
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    from document_loader import load_documents

    docs = load_documents()
    chunks = build_chunks(docs)

    print("文档数量：", len(docs))
    print("chunk 数量：", len(chunks))

    for chunk in chunks[:3]:
        print("=" * 80)
        print("source:", chunk["source"])
        print("chunk_id:", chunk["chunk_id"])
        print("chunk_length:", chunk["chunk_length"])
        print(chunk["text"][:300])