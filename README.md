
````markdown
# Security RAG Assistant

A lightweight Retrieval-Augmented Generation system for security knowledge base question answering and evaluation.

This project builds a local RAG pipeline over cybersecurity documents. It supports document loading, PDF text/table extraction, chunking, vector indexing, metadata filtering, hybrid retrieval, reranking, LLM-based answer generation, source tracing, refusal for out-of-scope questions, and basic evaluation.

The current version uses security documents such as NIST incident handling guidance, OWASP Web Top 10, and OWASP Top 10 for LLM Applications as example knowledge sources.

---

## Features

- Load `.txt`, `.md`, `.docx`, and text-based `.pdf` files.
- Extract PDF text and tables with `pdfplumber`.
- Convert extracted PDF tables into Markdown-style text.
- Split documents into chunks with a paragraph-first strategy.
- Build a local FAISS vector index with sentence-transformer embeddings.
- Retrieve relevant chunks with dense vector search.
- Support BM25 keyword retrieval and Hybrid Search.
- Use reranking to reorder retrieved candidates.
- Filter retrieval scope with document metadata categories.
- Generate answers using DeepSeek API.
- Display source document, chunk id, vector score, BM25 score, and rerank score.
- Reject questions when the knowledge base does not contain enough relevant information.
- Save query logs and evaluation outputs.
- Evaluate retrieval and answer quality with a small test set.

---

## Project Structure

```text
security-rag-v2/
├── app.py                  # Streamlit web interface
├── build_index.py           # Build FAISS index
├── config.py                # Global configuration
├── document_loader.py       # Document loading and PDF parsing
├── eval_rag.py              # Evaluation script
├── hybrid_search.py         # BM25 and hybrid retrieval
├── rag_pipeline.py          # Core RAG pipeline
├── text_splitter.py         # Text cleaning and chunking
├── requirements.txt
├── .env.example
│
├── data/
│   ├── .gitkeep
│   └── eval_questions.csv
│
├── index/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
├── notebooks/
│
└── screenshots/
````

---

## Environment Setup

Create a conda environment:

```bash
conda create -n security-rag python=3.10 -y
conda activate security-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
streamlit
sentence-transformers
faiss-cpu
numpy
pandas
python-dotenv
openai
tqdm
pypdf
pdfplumber
python-docx
FlagEmbedding
rank-bm25
jieba
```

---

## API Configuration

Create a `.env` file in the project root:

```bash
DEEPSEEK_API_KEY=your_api_key_here
```

The project provides `.env.example` as a template.

---

## Data Preparation

Place security documents under the `data/` directory.

Example documents used during development:

```text
data/
├── nist.sp.800-61r2.pdf
├── owasp-top-10.pdf
├── OWASP-Top-10-for-LLMs-v2025.pdf
└── eval_questions.csv
```

The current metadata categories are configured in `config.py`:

```python
DOCUMENT_CATEGORIES = {
    "nist.sp.800-61r2.pdf": "incident_response",
    "owasp-top-10.pdf": "web_security",
    "OWASP-Top-10-for-LLMs-v2025.pdf": "llm_security"
}
```

These categories are used for metadata filtering during retrieval.

---

## Build the Index

Run:

```bash
python build_index.py
```

This step will:

1. Load documents from `data/`.
2. Extract text and tables.
3. Split documents into chunks.
4. Generate embeddings.
5. Build a FAISS index.
6. Save generated files under `index/`.

Generated files include:

```text
index/faiss.index
index/chunks.json
index/index_meta.json
```

---

## Run the Web App

Start the Streamlit interface:

```bash
streamlit run app.py
```

The app provides:

* Question input;
* Knowledge scope selection;
* Rerank switch;
* Hybrid Search switch;
* Top-K settings;
* Similarity threshold setting;
* Answer display;
* Retrieved chunk display;
* Source file, chunk id, vector score, BM25 score, and rerank score.

---

## Run Evaluation

Run:

```bash
python eval_rag.py
```

Evaluation outputs are saved under `logs/`:

```text
logs/eval_results.csv
logs/eval_summary.json
logs/threshold_compare.csv
logs/rerank_compare.csv
logs/query_logs.jsonl
```

The evaluation set is stored in:

```text
data/eval_questions.csv
```

Example format:

```csv
question,expected_keywords,expected_source,should_answer
What are the major phases of the incident response life cycle?,Preparation;Detection and Analysis;Containment;Recovery,nist.sp.800-61r2.pdf,yes
What is Prompt Injection in LLM applications?,Prompt Injection;Direct Prompt Injections;Indirect Prompt Injections,OWASP-Top-10-for-LLMs-v2025.pdf,yes
What is the weather in Singapore tomorrow?,,none,no
```

---

## System Pipeline

```text
Documents
  ↓
Document loading and PDF parsing
  ↓
Text cleaning and table-to-Markdown conversion
  ↓
Chunk splitting
  ↓
Embedding generation
  ↓
FAISS index construction
  ↓
User query
  ↓
Metadata filtering
  ↓
Vector retrieval / BM25 retrieval / Hybrid Search
  ↓
Reranking
  ↓
Similarity threshold check
  ↓
Prompt construction
  ↓
DeepSeek answer generation
  ↓
Answer, sources, and logs
```

---

## Current Evaluation Results

The current evaluation set contains 17 questions:

* 13 answerable questions;
* 4 unanswerable questions.

Evaluation result:

```json
{
  "total_questions": 17,
  "answerable_questions": 13,
  "unanswerable_questions": 4,
  "avg_best_score_answerable": 0.7451,
  "source_hit_rate": 0.9231,
  "top1_source_hit_rate": 0.8462,
  "avg_keyword_hit_rate": 0.6115,
  "refusal_accuracy": 1.0
}
```

Metric explanation:

| Metric                 | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `source_hit_rate`      | Whether the expected source document appears in Top-K results |
| `top1_source_hit_rate` | Whether the Top-1 result is from the expected source document |
| `avg_keyword_hit_rate` | Average keyword coverage in generated answers                 |
| `refusal_accuracy`     | Accuracy of refusing out-of-knowledge-base questions          |

The results show that the system can retrieve relevant sources from real English security PDFs and can reject out-of-scope questions after threshold tuning.

---

## Rerank Experiment

A rerank comparison was conducted to evaluate whether reranking improves retrieval ordering.

```text
use_rerank=False
top1_source_hit_rate = 0.8462
avg_keyword_hit_rate = 0.5756

use_rerank=True
top1_source_hit_rate = 0.9231
avg_keyword_hit_rate = 0.5962
```

In the current evaluation set, enabling rerank improves both Top-1 source hit rate and average keyword hit rate.

Note that `rerank_score` is not a probability or cosine similarity. It is an internal relevance score produced by the reranker model, and its relative ordering is more important than its absolute value.

---

## Threshold Experiment

The system uses `best_score` to decide whether the retrieved context is sufficiently relevant.

The vector index uses normalized embeddings with FAISS inner product search:

```python
normalize_embeddings=True
faiss.IndexFlatIP
```

Therefore, the vector score can be interpreted approximately as cosine similarity.

During evaluation, different similarity thresholds were tested. A lower threshold makes the system more likely to answer, but may increase incorrect answers for unrelated questions. A higher threshold makes the system more conservative.

In the current setup, the default threshold is:

```python
SIMILARITY_THRESHOLD = 0.40
```

This setting improves refusal behavior for out-of-scope questions.

---

## Key Design Notes

### Metadata Filtering

When all documents are searched together, ambiguous terms may cause retrieval confusion.

For example, the term `Injection` may refer to:

* Web security Injection in OWASP Web Top 10;
* Prompt Injection in OWASP LLM Top 10.

To reduce this ambiguity, each document is assigned a metadata category:

```text
incident_response
web_security
llm_security
```

The app allows users to choose the retrieval scope. This helps the system retrieve from the intended knowledge domain.

An implementation issue was found during development: filtering after retrieving only a small global Top-K could remove all relevant chunks from the selected category. The retrieval logic was updated to expand candidate retrieval first, then apply metadata filtering.

### PDF Table Extraction

Some security PDFs contain important tables. Basic PDF text extraction may lose row-column relationships, which can reduce answer completeness.

To improve this, the loader uses `pdfplumber` to extract tables and convert them into Markdown-style text before indexing.

This helps preserve table information, but complex PDF layouts may still require more advanced parsing.

### Hybrid Search

Dense vector retrieval is useful for semantic matching, while BM25 is useful for exact keyword matching.

Security documents often contain terms such as:

```text
CWE-89
SSRF
SQL Injection
LLM08
NIST 800-61
```

Hybrid Search combines vector retrieval and BM25 retrieval before reranking. This improves retrieval robustness for both semantic queries and exact technical terms.

---

## Limitations

The current system is still a lightweight prototype. Main limitations include:

1. PDF parsing is limited for complex tables, scanned PDFs, images, and multi-column layouts.
2. Chunking is based mainly on paragraphs and does not fully use document section hierarchy.
3. Hybrid Search currently merges candidates but does not perform advanced score fusion.
4. Keyword hit rate is a simple evaluation metric and cannot fully capture semantic correctness.
5. The system mainly supports single-turn question answering.
6. No user permission or document-level access control is implemented.
7. Index updates require rebuilding the index manually.

---

## Future Work

Planned improvements include:

* Better PDF table and layout parsing;
* Section-aware chunking;
* Page number and section title metadata;
* More robust Hybrid Search fusion;
* Query rewriting;
* Manual failure reason annotation;
* Groundedness and answer relevance evaluation;
* Multi-turn conversation support;
* Document-level permission control;
* Log sanitization.

---

## What This Project Demonstrates

This project demonstrates the following RAG capabilities:

* Building a local knowledge base from real security documents;
* Implementing the full RAG pipeline from ingestion to answer generation;
* Using FAISS for vector retrieval;
* Applying rerank for retrieval refinement;
* Using metadata filtering to reduce domain ambiguity;
* Using Hybrid Search to improve keyword-sensitive retrieval;
* Adding similarity thresholding to reduce unsupported answers;
* Evaluating RAG behavior with source hit rate, keyword hit rate, and refusal accuracy;
* Analyzing common failure cases in realistic PDF-based RAG systems.

---

## Resume Description

Built a lightweight cybersecurity RAG question-answering and evaluation system using Python, FAISS, BM25, Rerank, Streamlit, and DeepSeek API. The system supports txt/md/docx/text-based PDF ingestion, PDF text and table extraction, paragraph-first chunking, vector indexing, metadata filtering, hybrid retrieval, reranking, source tracing, and threshold-based refusal for out-of-knowledge-base questions. Evaluated the system on real English security PDFs from NIST and OWASP using source hit rate, Top-1 source hit rate, keyword hit rate, refusal accuracy, and rerank comparison experiments.

```
```
