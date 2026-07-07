import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import (
    CHUNK_STRATEGY_VERSION,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    DOCUMENT_CATEGORIES,
    EMBEDDING_MODEL_NAME,
    MIN_CHUNK_TOKENS,
    PARENT_TOKEN_OVERLAP,
    PARENT_TOKEN_SIZE,
)


TokenSpan = Tuple[int, int]


SECTION_TYPE_ALIASES = {
    "abstract": "abstract",
    "summary": "abstract",
    "summary of results": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "method overview": "method",
    "approach": "method",
    "model": "method",
    "experiments": "experiments",
    "experiment": "experiments",
    "evaluation": "evaluation",
    "results": "results",
    "analysis": "analysis",
    "discussion": "discussion",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "future work": "future_work",
    "appendix": "appendix",
    "references": "references",
    "acknowledgments": "back_matter",
    "acknowledgements": "back_matter",
}

PAPER_SECTION_MARKERS = [
    "Abstract",
    "Summary of Results",
    "Introduction",
    "Background",
    "Related Work",
    "Method Overview",
    "Method",
    "Methods",
    "Methodology",
    "Approach",
    "Experiments",
    "Evaluation",
    "Results",
    "Analysis",
    "Discussion",
    "Limitations",
    "Conclusion",
    "Conclusions",
    "Future Work",
    "Appendix",
    "References",
    "Acknowledgments",
    "Acknowledgements",
    "Using Sparse Autoencoders To Find Good Decompositions",
    "Feature Splitting",
    "Universality",
    "Automated Interpretability",
    "Advice for Training Sparse Autoencoders",
    "Circuit Tracing",
    "Graph Pruning",
    "Multi-step Reasoning",
    "Planning in Poems",
    "Multilingual Circuits",
    "Medical Diagnoses",
    "Entity Recognition and Hallucinations",
    "Refusals",
    "Life of a Jailbreak",
    "Chain-of-thought Faithfulness",
    "Uncovering Hidden Goals",
    "Commonly Observed Circuit Components and Structure",
    "Open Questions",
]

BOILERPLATE_LINES = {
    "×",
    "Transformer Circuits Thread",
    "Citation Information",
    "Author Contributions",
}


@dataclass
class TextBlock:
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Section:
    index: int
    title: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_type: str = "body"


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph and page boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document_title(text: str, fallback: str = "Document") -> str:
    for raw_line in clean_text(text).splitlines()[:40]:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            if title:
                return title
        if not line.lower().startswith(("source:", "downloaded:")):
            return line[:180]
    return fallback


def _normalize_heading_key(title: str) -> str:
    title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title).strip()
    title = re.sub(r"\s+", " ", title)
    return title.lower().strip(" .:-")


def classify_section(title: str) -> str:
    key = _normalize_heading_key(title)
    for marker, section_type in SECTION_TYPE_ALIASES.items():
        if key == marker or key.startswith(marker + " "):
            return section_type
    if key.startswith("appendix"):
        return "appendix"
    if "table" in key:
        return "table"
    if "figure" in key:
        return "figure"
    return "body"


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or line in BOILERPLATE_LINES:
        return False
    if line.startswith(("|", "[Table", "[Figure")):
        return False
    if re.fullmatch(r"\[Page \d+\]", line):
        return False
    if len(line) > 150:
        return False

    key = _normalize_heading_key(line)
    if key in SECTION_TYPE_ALIASES:
        return True
    if any(key == marker.lower() for marker in PAPER_SECTION_MARKERS):
        return True
    if any(key.startswith(marker.lower() + ":") for marker in PAPER_SECTION_MARKERS):
        return True

    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,:;'\-/()&]{2,130}$", line):
        return not line.endswith(".")

    alpha_chars = [ch for ch in line if ch.isalpha()]
    if 3 <= len(line) <= 90 and alpha_chars:
        upper_ratio = sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)
        if upper_ratio > 0.75 and not line.endswith("."):
            return True

    return False


def _insert_inline_heading_breaks(line: str) -> List[str]:
    if len(line) < 180:
        return [line]

    updated = line
    markers = sorted(PAPER_SECTION_MARKERS, key=len, reverse=True)
    for marker in markers:
        pattern = rf"(?<!^)(?<![A-Za-z])({re.escape(marker)})(?=\s+[A-Z0-9])"
        updated = re.sub(pattern, r"\n\1\n", updated)

    return [part.strip() for part in updated.splitlines() if part.strip()]


def _block_text(lines: Sequence[str]) -> str:
    structured = any(
        line.startswith(("|", "-", "*", ">", "[Table", "[Figure"))
        or re.match(r"^\d+[\).\s]", line)
        for line in lines
    )
    if structured:
        return "\n".join(lines).strip()
    return " ".join(lines).strip()


def _iter_blocks(text: str) -> Iterable[TextBlock]:
    current_lines: List[str] = []
    current_page: Optional[int] = None
    block_page_start: Optional[int] = None

    def flush() -> Optional[TextBlock]:
        nonlocal current_lines, block_page_start
        if not current_lines:
            return None
        block = TextBlock(
            text=_block_text(current_lines),
            page_start=block_page_start,
            page_end=current_page,
        )
        current_lines = []
        block_page_start = None
        return block

    for raw_line in clean_text(text).splitlines():
        stripped = raw_line.strip()

        page_match = re.fullmatch(r"\[Page (\d+)\]", stripped)
        if page_match:
            block = flush()
            if block and block.text:
                yield block
            current_page = int(page_match.group(1))
            continue

        if not stripped:
            block = flush()
            if block and block.text:
                yield block
            continue

        if stripped.lower().startswith(("title:", "source:", "downloaded:")):
            continue

        for line in _insert_inline_heading_breaks(stripped):
            if line in BOILERPLATE_LINES:
                continue
            if _looks_like_heading(line):
                block = flush()
                if block and block.text:
                    yield block
                yield TextBlock(text=line, page_start=current_page, page_end=current_page)
                continue

            if block_page_start is None:
                block_page_start = current_page
            current_lines.append(line)

    block = flush()
    if block and block.text:
        yield block


def split_into_sections(text: str, fallback_title: str = "Document") -> List[Section]:
    sections: List[Section] = []
    current_title = "Front Matter"
    current_blocks: List[TextBlock] = []
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None

    def flush() -> None:
        nonlocal current_blocks, current_page_start, current_page_end
        body = "\n\n".join(block.text for block in current_blocks if block.text).strip()
        if body:
            sections.append(
                Section(
                    index=len(sections),
                    title=current_title or fallback_title,
                    text=body,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    section_type=classify_section(current_title),
                )
            )
        current_blocks = []
        current_page_start = None
        current_page_end = None

    for block in _iter_blocks(text):
        if _looks_like_heading(block.text):
            flush()
            current_title = block.text
            current_page_start = block.page_start
            current_page_end = block.page_end
            continue

        if current_page_start is None:
            current_page_start = block.page_start
        if block.page_end is not None:
            current_page_end = block.page_end
        current_blocks.append(block)

    flush()

    if not sections:
        cleaned = clean_text(text)
        if cleaned:
            sections.append(
                Section(
                    index=0,
                    title=fallback_title,
                    text=cleaned,
                    section_type="body",
                )
            )

    return sections


@lru_cache(maxsize=1)
def _load_fast_tokenizer():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_NAME,
            local_files_only=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "prepare_for_model"):
            tokenizer.model_max_length = 10**9
            return tokenizer
    except Exception:
        return None
    return None


def _token_spans(text: str) -> List[TokenSpan]:
    tokenizer = _load_fast_tokenizer()
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        spans = [
            (int(start), int(end))
            for start, end in encoded.get("offset_mapping", [])
            if end > start
        ]
        if spans:
            return spans

    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def token_count(text: str) -> int:
    return len(_token_spans(text))


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> List[Dict]:
    text = text.strip()
    if not text:
        return []

    spans = _token_spans(text)
    if not spans:
        return []

    max_tokens = max(1, int(max_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), max_tokens - 1))

    if len(spans) <= max_tokens:
        if len(spans) >= min_tokens:
            return [{"text": text, "token_count": len(spans), "token_start": 0, "token_end": len(spans)}]
        return []

    chunks = []
    start = 0
    while start < len(spans):
        end = min(start + max_tokens, len(spans))
        char_start = spans[start][0]
        char_end = spans[end - 1][1]
        chunk_text = text[char_start:char_end].strip()

        if chunk_text and end - start >= min_tokens:
            chunks.append(
                {
                    "text": chunk_text,
                    "token_count": end - start,
                    "token_start": start,
                    "token_end": end,
                }
            )

        if end >= len(spans):
            break
        start = max(start + 1, end - overlap_tokens)

    return chunks


def _stable_parent_id(doc_key: str, section_index: int, parent_index: int) -> str:
    raw = f"{doc_key}:{section_index}:{parent_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _page_label(page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start == page_end or page_end is None:
        return f"Page: {page_start}"
    return f"Pages: {page_start}-{page_end}"


def _context_prefix(doc: Dict, paper_title: str, section: Section) -> str:
    parts = [
        f"Paper: {paper_title}",
        f"Source: {doc.get('source', '')}",
        f"Section: {section.title}",
    ]
    page_label = _page_label(section.page_start, section.page_end)
    if page_label:
        parts.append(page_label)
    return "\n".join(parts)


def _with_prefix(prefix: str, text: str) -> str:
    return f"{prefix}\n\n{text.strip()}".strip()


def split_text_by_paragraph(
    text: str,
    chunk_size: int = CHUNK_TOKEN_SIZE,
    overlap: int = CHUNK_TOKEN_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_TOKENS,
) -> List[str]:
    """Compatibility wrapper returning token-based section chunks as strings."""
    title = extract_document_title(text)
    outputs: List[str] = []
    for section in split_into_sections(text, fallback_title=title):
        for part in split_text_by_tokens(section.text, chunk_size, overlap, min_chunk_size):
            outputs.append(part["text"])
    return outputs


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """Build child retrieval chunks with larger parent context metadata."""
    all_chunks: List[Dict] = []

    for doc in documents:
        paper_title = extract_document_title(doc["text"], fallback=doc["source"])
        doc_key = doc.get("doc_id") or doc.get("relative_path") or doc["source"]
        sections = split_into_sections(doc["text"], fallback_title=paper_title)
        chunk_id = 0

        for section in sections:
            prefix = _context_prefix(doc, paper_title, section)
            parent_parts = split_text_by_tokens(
                section.text,
                max_tokens=PARENT_TOKEN_SIZE,
                overlap_tokens=PARENT_TOKEN_OVERLAP,
                min_tokens=MIN_CHUNK_TOKENS,
            )

            for parent_index, parent_part in enumerate(parent_parts):
                parent_id = _stable_parent_id(doc_key, section.index, parent_index)
                parent_text = _with_prefix(prefix, parent_part["text"])
                child_parts = split_text_by_tokens(
                    parent_part["text"],
                    max_tokens=CHUNK_TOKEN_SIZE,
                    overlap_tokens=CHUNK_TOKEN_OVERLAP,
                    min_tokens=MIN_CHUNK_TOKENS,
                )

                for child_index, child_part in enumerate(child_parts):
                    child_text = _with_prefix(prefix, child_part["text"])
                    item = {
                        "source": doc["source"],
                        "path": doc["path"],
                        "relative_path": doc.get("relative_path", doc["source"]),
                        "extension": doc["extension"],
                        "category": doc.get("category", DOCUMENT_CATEGORIES.get(doc["source"], "general")),
                        "paper_title": paper_title,
                        "chunk_id": chunk_id,
                        "chunk_length": len(child_text),
                        "chunk_token_count": child_part["token_count"],
                        "chunk_strategy": CHUNK_STRATEGY_VERSION,
                        "text": child_text,
                        "retrieval_text": child_text,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "parent_length": len(parent_text),
                        "parent_token_count": parent_part["token_count"],
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "section_index": section.index,
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    }
                    if doc.get("doc_id"):
                        item["doc_id"] = doc["doc_id"]
                        item["chunk_uid"] = f"{doc['doc_id']}:{chunk_id}"
                    if doc.get("file_hash"):
                        item["file_hash"] = doc["file_hash"]
                    all_chunks.append(item)
                    chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    from document_loader import load_documents

    docs = load_documents()
    chunks = build_chunks(docs)

    print("documents:", len(docs))
    print("chunks:", len(chunks))

    for chunk in chunks[:3]:
        print("=" * 80)
        print("source:", chunk["source"])
        print("chunk_id:", chunk["chunk_id"])
        print("section:", chunk.get("section_title"))
        print("parent_id:", chunk.get("parent_id"))
        print("chunk_tokens:", chunk.get("chunk_token_count"))
        print(chunk["text"][:500])
