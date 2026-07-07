# AI Research Paper RAG Demo

AI Research Paper RAG Demo is a local Retrieval-Augmented Generation project for reading, indexing, evaluating, and querying AI research papers. The current corpus focuses on mechanistic interpretability, sparse autoencoders, deception, alignment faking, sycophancy, jailbreak evaluation, dangerous capability evaluation, and chain-of-thought oversight.

The project is designed as a learning and portfolio POC: small enough to understand, but complete enough to show document ingestion, paper-aware parent-child chunking, FAISS indexing, incremental vector updates, retrieval, reranking, answer generation, Streamlit UI, Agent-style routing, MCP tool integration, retrieval evaluation, and RAGAS answer-level evaluation.

## Highlights

- Local paper RAG over PDF, TXT, Markdown, and DOCX files
- Current paper corpus under `data/papers/` with 23 AI interpretability, alignment, deception, and safety evaluation sources
- Local embedding model support with `BAAI/bge-m3`
- FAISS vector store using `IndexIDMap2(IndexFlatIP)` for stable vector ids
- Incremental index update through `index_manifest.json`
- Optional BM25 + vector hybrid retrieval
- Optional reranking with `BAAI/bge-reranker-base`
- Streamlit UI for Normal RAG and Agent RAG modes
- MCP stdio server for tool-based access from MCP hosts
- Retrieval evaluation with 30 questions, graded relevance, negative queries, noise metrics, and grouped summaries

## Architecture

```text
User / Streamlit UI
-> app.py
-> RAGPipeline
-> FAISS / optional BM25 / optional reranker
-> DeepSeek Chat
-> grounded answer + sources
```

```text
MCP Client
-> stdio MCP protocol
-> mcp_server_sdk.py
-> rag_service.py
-> cached RAGPipeline / cached AgentRAGWorkflow
-> retrieval / answer / agent answer
-> structured MCP tool result
```

MCP does not replace the RAG pipeline. It wraps the same local RAG and Agent RAG capabilities behind a standard tool interface, so MCP hosts such as Claude Desktop, Cursor, Codex, or other clients can call the project logic.

## Tech Stack

- Python 3.10
- Conda environment: `rag`
- Streamlit
- FAISS
- sentence-transformers
- `BAAI/bge-m3` embeddings
- `BAAI/bge-reranker-base` reranker
- rank-bm25
- FlagEmbedding
- MCP Python SDK / FastMCP
- DeepSeek API
- pandas / numpy
- pdfplumber / python-docx

## Project Structure

```text
rag-demo/
|-- app.py                         # Streamlit app: Normal RAG and Agent RAG
|-- api_server.py                  # FastAPI backend for RAG, Agent RAG, document upload, and index status
|-- auth.py                        # Simple API key authentication dependency
|-- agent_router.py                # Query Router + Tool Calling style Agent workflow
|-- build_index.py                 # Compatibility entrypoint for full index rebuild
|-- config.py                      # Paths, chunking, retrieval, model, and LLM config
|-- document_loader.py             # PDF/TXT/Markdown/DOCX document loading
|-- embedding_utils.py             # Embedding device resolution: auto GPU with CPU fallback
|-- eval_rag.py                    # Legacy answer-level evaluation script
|-- eval_retrieval.py              # Retrieval evaluation script
|-- frontend_app.py                # Streamlit frontend that calls the FastAPI backend
|-- hybrid_search.py               # BM25 and hybrid search helpers
|-- index_manager.py               # Incremental FAISS IndexIDMap2 index management
|-- index_tasks.py                 # Background indexing task runner with in-process write lock
|-- rag_pipline.py                 # Core RAG pipeline
|-- rag_service.py                 # Service layer used by MCP tools
|-- schemas.py                     # Pydantic request/response schemas
|-- service_context.py             # Shared logging, timing, and env snapshots
|-- storage.py                     # Safe document upload storage and upload manifest helpers
|-- mcp_server_sdk.py              # Main FastMCP stdio server
|-- mcp_server.py                  # Manual stdio MCP server for protocol debugging
|-- mcp_client_demo.py             # Local MCP client demo
|-- text_splitter.py               # Chunk cleaning and splitting logic
|-- requirements.txt
|-- .env.example
|-- data/
|   |-- papers/                    # AI research paper corpus
|   |-- retrieval_eval_questions.csv
|   `-- eval_questions.csv         # Legacy evaluation questions
|-- docs/                          # MCP design notes and development log
|-- index/                         # Generated FAISS/chunks/manifest files; ignored by Git
|-- logs/                          # Local generated logs and eval outputs; ignored by Git
`-- models/                        # Local embedding/rerank model cache; ignored by Git
```

## Quick Start

This project uses the Conda environment named `rag`.

```powershell
conda activate rag
pip install -r requirements.txt
```

If you need to create the environment on a new machine:

```powershell
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
APP_API_KEY=change-me
EMBEDDING_DEVICE=auto
MAX_UPLOAD_BYTES=26214400
```

The `.env` file is ignored by Git and should never be committed.

## FastAPI Backend and API Frontend

The project now includes an HTTP backend that reuses `rag_service.py` instead of duplicating RAG logic inside the web layer.

Start the FastAPI backend:

```powershell
conda activate rag
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health check |
| `GET /index/status` | FAISS/chunks/manifest status |
| `GET /sources` | Indexed knowledge sources |
| `GET /documents` | Uploaded document records from `data/uploads/upload_manifest.json` |
| `POST /documents/upload` | Upload a PDF/TXT/MD/DOCX, save it safely, queue indexing, and update status |
| `POST /chat` | Normal RAG answer |
| `POST /agent/chat` | Agent RAG answer |

If `APP_API_KEY` is set to a real value, protected endpoints expect:

```text
Authorization: Bearer <APP_API_KEY>
```

Run the API-backed Streamlit frontend:

```powershell
conda activate rag
streamlit run frontend_app.py
```

The original `app.py` is still kept as a local debug UI that imports `RAGPipeline` directly. `frontend_app.py` is the engineering-style UI that calls the FastAPI backend over HTTP.

## Local Embedding Model

The current embedding model is configured in `config.py`:

```python
EMBEDDING_MODEL_NAME = os.path.join(BASE_DIR, "models", "embeddings", "bge-m3")
```

Expected local folder:

```text
models/embeddings/bge-m3/
```

If the model is not already present, download it from Hugging Face:

```powershell
huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/bge-m3
```

After the model is cached locally, offline mode avoids slow Hugging Face network checks:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

For RTX 50-series GPUs, use a PyTorch build that supports `sm_120`. This workspace has been tested with CUDA-enabled PyTorch and GPU encoding for rebuilds.

## Corpus

The current main corpus lives in:

```text
data/papers/
```

It includes papers and official research pages covering:

- Deception, alignment faking, sleeper agents, sycophancy, and truth representations
- Mechanistic interpretability, sparse autoencoders, monosemantic features, circuits, and benchmarks
- Attention and saliency explanation reliability
- Jailbreak benchmarks, constitutional classifiers, chain-of-thought monitorability, and dangerous capability evaluations

Original security documents may still exist under `data/` as legacy sample sources, but the current UI and evaluation focus on AI research papers.

## Chunking Logic

Chunking is implemented in `text_splitter.py`.

Current settings in `config.py`:

```python
CHUNK_STRATEGY_VERSION = "paper_section_token_parent_v1"
CHUNK_TOKEN_SIZE = 450
CHUNK_TOKEN_OVERLAP = 80
MIN_CHUNK_TOKENS = 40
PARENT_TOKEN_SIZE = 1500
PARENT_TOKEN_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_MAX_SEQ_LENGTH = 768
```

Current behavior:

1. `document_loader.py` extracts text from PDF/TXT/Markdown/DOCX files.
2. PDF pages are prefixed with markers such as `[Page 1]`; tables are converted to Markdown-style tables.
3. `clean_text()` normalizes line endings, de-hyphenates broken PDF line wraps, compresses repeated blank lines, and removes repeated spaces.
4. `split_into_sections()` detects paper-like headings such as Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Limitations, Appendix, and References. It also handles long Transformer Circuits web-page lines by inserting section breaks around known headings.
5. Each section is split into larger parent contexts of about `PARENT_TOKEN_SIZE` tokens with `PARENT_TOKEN_OVERLAP`.
6. Each parent is split into child retrieval chunks of about `CHUNK_TOKEN_SIZE` tokens with `CHUNK_TOKEN_OVERLAP`.
7. FAISS embeds the child chunk text. At answer time, the pipeline promotes matched children back to their parent context before building the LLM prompt.
8. `build_chunks()` attaches metadata such as `paper_title`, `section_title`, `section_type`, `page_start`, `page_end`, `parent_id`, `parent_text`, `doc_id`, `chunk_uid`, `file_hash`, `chunk_id`, `chunk_token_count`, and `parent_token_count`.

This means retrieval stays precise while answer generation receives enough surrounding context to handle paper summaries, comparisons, and synthesis questions.

Known limitations:

- PDF extraction can still produce noisy headings when a paper has complex columns, figures, or tables.
- References and appendix content are indexed, but they are marked by section metadata rather than excluded.
- Legacy security sample PDFs under `data/` are still indexed unless removed or moved out of the data folder.

Changing chunk settings changes the vector contents. Run a full rebuild after changing chunking logic:

```powershell
conda run -n rag python index_manager.py rebuild
```

## Build and Update the Index

Run this after the first clone or after major settings changes:

```powershell
conda activate rag
python build_index.py
```

`build_index.py` is a compatibility entrypoint. It delegates to `index_manager.py rebuild`.

For daily document changes:

```powershell
conda run -n rag python index_manager.py scan
conda run -n rag python index_manager.py update
conda run -n rag python index_manager.py status
```

If embedding, chunking, or index settings change, run a full rebuild:

```powershell
conda run -n rag python index_manager.py rebuild
```

To remove one indexed document from the vector store without deleting the source file:

```powershell
conda run -n rag python index_manager.py remove --source papers/example.pdf
```

Generated files:

```text
index/faiss.index
index/chunks.json
index/index_meta.json
index/index_manifest.json
```

These files are local generated artifacts and are ignored by Git.

## Incremental Indexing

`index_manager.py` uses `index/index_manifest.json` to track document hashes, vector ids, and index settings.

The manifest records:

- `doc_id`
- `relative_path`
- `file_hash`
- `chunk_count`
- `vector_ids`
- embedding model
- chunk settings
- FAISS index type
- embedding dimension

When you run `update`, the manager:

1. Scans supported files under `data/`.
2. Compares current file hashes with the manifest.
3. Removes vector ids for deleted or modified documents.
4. Re-chunks and re-embeds only added or modified documents.
5. Adds new vectors with stable explicit ids through FAISS `IndexIDMap2`.

If the embedding model or chunking settings no longer match the manifest, `update` stops and asks for `rebuild`.

## Run the Streamlit App

```powershell
conda activate rag
streamlit run app.py
```

Offline local-model mode:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
conda run -n rag python -m streamlit run app.py
```

The app title is `AI Research Paper RAG Demo`. It includes example questions about alignment faking, Sleeper Agents, sparse autoencoders, and mechanistic interpretability.

## App Modes

### Normal RAG

Normal RAG uses one fixed chain:

```text
User Query -> Retrieval -> Rerank -> Prompt -> DeepSeek -> Answer + Sources
```

It is suitable for direct paper QA, such as:

```text
How do sparse autoencoders help mechanistic interpretability?
Compare alignment faking and Sleeper Agents.
What does Chain of Thought Monitorability claim is fragile?
```

### Agent RAG

Agent RAG adds a lightweight Query Router and tool selection layer:

```text
User Query -> Query Router -> Tool Selection -> Retrieval/Prompt -> Answer + Sources
```

The router uses rule-based classification instead of an LLM classifier.

| Task type | Purpose |
| --- | --- |
| `fact_qa` | Direct factual paper QA |
| `table_qa` | Table, field, column, or structured metadata questions |
| `summary` | Summary or overview questions |
| `compare` | Comparison or difference questions |

Agent tools implemented in `agent_router.py`:

| Tool | Purpose |
| --- | --- |
| `search_docs(query)` | Standard document retrieval QA |
| `search_tables(query)` | Prefer table-like or field-like chunks, then fall back to normal retrieval |
| `summarize_docs(query)` | Retrieve more context and use a summary prompt |
| `compare_docs(query)` | Retrieve more context and use a comparison prompt |

## MCP Integration

The main MCP server is:

```text
mcp_server_sdk.py
```

It uses FastMCP and stdio transport. The manual server, `mcp_server.py`, is kept as a protocol debugging reference.

Some MCP tool names still contain `security` for backwards compatibility with earlier versions. The underlying corpus and UI now focus on AI research papers.

### MCP Tools

| Tool | Purpose |
| --- | --- |
| `get_index_status` | Check whether FAISS index, chunks, and metadata exist; return document and embedding metadata |
| `list_knowledge_sources` | List indexed document sources and categories |
| `warmup_rag` | Preload the embedding model, FAISS index, chunks, and optional Agent workflow |
| `benchmark_retrieval` | Run retrieval twice in one server session and compare first/second call latency |
| `retrieve_security_chunks` | Retrieve relevant chunks without calling the LLM |
| `answer_security_question` | Answer with the normal RAG pipeline and return answer plus sources |
| `agent_security_answer` | Answer with Agent RAG through Query Router + tool selection |

### Run the Local MCP Demo

Lightweight mode lists tools and checks index/source metadata. It does not load the embedding model or FAISS index:

```powershell
conda run -n rag python mcp_client_demo.py
```

Warm up the RAG pipeline without running a retrieval query:

```powershell
conda run -n rag python mcp_client_demo.py --warmup
```

Run a real retrieval tool call:

```powershell
conda run -n rag python mcp_client_demo.py --with-retrieval
```

Benchmark cached retrieval in the same MCP server session:

```powershell
conda run -n rag python mcp_client_demo.py --benchmark
```

### MCP Host Configuration Example

For an MCP host that accepts stdio server configuration, use the Conda `rag` environment and point the command at `mcp_server_sdk.py`:

```json
{
  "mcpServers": {
    "ai-research-rag-mcp": {
      "command": "conda",
      "args": [
        "run",
        "-n",
        "rag",
        "python",
        "<absolute-path-to-rag-demo>\\mcp_server_sdk.py"
      ],
      "cwd": "<absolute-path-to-rag-demo>",
      "env": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false"
      }
    }
  }
}
```

Important MCP note: stdio MCP uses stdout as the protocol channel. The MCP server redirects normal prints and model-loading messages to stderr so JSON-RPC frames are not polluted.

## Retrieval Evaluation

Retrieval evaluation questions are stored in:

```text
data/retrieval_eval_questions.csv
```

The current evaluation set contains 30 questions with:

- positive and negative queries
- easy, medium, and hard difficulty labels
- query types such as fact, paraphrase, compare, synthesis, keyword, and negative
- primary expected sources
- partial sources for graded relevance
- expected keywords

Run the default retrieval evaluation:

```powershell
conda run -n rag python eval_retrieval.py --top-k 5 --candidate-k 20
```

Compare hybrid retrieval:

```powershell
conda run -n rag python eval_retrieval.py --top-k 5 --candidate-k 20 --hybrid
```

Compare reranking:

```powershell
conda run -n rag python eval_retrieval.py --top-k 5 --candidate-k 20 --rerank
```

Outputs:

```text
logs/retrieval_eval_results.csv
logs/retrieval_eval_summary.json
```

The script reports metrics such as:

- hit rate
- top-1 primary and graded accuracy
- primary source recall
- graded recall
- chunk precision
- MRR
- nDCG
- keyword hit rate
- final and candidate noise rate
- negative high-confidence false positive rate
- summaries by query type and difficulty

This is still retrieval-only evaluation. It does not yet judge final answer faithfulness, citation correctness, or hallucination.

## Answer-Level RAGAS Evaluation

`eval_ragas.py` evaluates generated answers with RAGAS using the same question CSV:

```text
data/retrieval_eval_questions.csv
```

The CSV includes `reference_answer` for answerable questions, so the default RAGAS metrics are:

```text
faithfulness
answer_relevancy
context_precision
context_recall
answer_correctness
```

Run Normal RAG answer evaluation:

```powershell
conda run -n rag python eval_ragas.py
```

Run Agent RAG answer evaluation, including query-type routing for compare, synthesis, and literature-review questions:

```powershell
conda run -n rag python eval_ragas.py --agent
```

Outputs:

```text
logs/ragas_eval_dataset.jsonl
logs/ragas_eval_results.csv
logs/ragas_eval_summary.json
```

## Legacy Answer Evaluation

`eval_rag.py` and `data/eval_questions.csv` are kept for earlier answer-level experiments.

```powershell
conda run -n rag python eval_rag.py
```

Use `eval_retrieval.py` for the current retrieval-focused evaluation workflow.

## Logs

| File | Purpose |
| --- | --- |
| `logs/query_logs.jsonl` | Normal RAG query logs |
| `logs/agent_log.jsonl` | Agent RAG debug logs |
| `logs/service_perf.log` | Service-layer timing and environment snapshots |
| `logs/mcp_client_demo_stderr.log` | MCP server stderr captured by the local demo client |
| `logs/retrieval_eval_results.csv` | Per-question retrieval evaluation output |
| `logs/retrieval_eval_summary.json` | Summary retrieval evaluation metrics |

## Useful Development Commands

Syntax check:

```powershell
conda run -n rag python -m py_compile service_context.py rag_service.py rag_pipline.py mcp_server_sdk.py mcp_server.py mcp_client_demo.py eval_retrieval.py index_manager.py
```

Test retrieval directly without MCP:

```powershell
conda run -n rag python -c "from rag_service import retrieve_security_chunks; import json; result=retrieve_security_chunks('How do sparse autoencoders help mechanistic interpretability?', top_k=3, use_rerank=False, include_text=False); print(json.dumps(result, ensure_ascii=False, indent=2))"
```

View logs on Windows:

```powershell
Get-Content logs\service_perf.log
Get-Content logs\mcp_client_demo_stderr.log
```

## Performance Notes

Slow startup is usually caused by:

- Loading the SentenceTransformer embedding model
- Loading FAISS index and `chunks.json`
- Initializing BM25
- Loading the optional reranker model
- Hugging Face cache checks
- First import of scientific dependencies inside Windows + FastMCP tool calls

Useful optimizations:

- Use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` after models are cached
- Set `USE_RERANK = False` in `config.py` for faster demos
- Use `warmup_rag` before real MCP retrieval or answer calls
- Use `retrieve_security_chunks` for grounding checks that do not need an LLM answer
- Keep the cached `RAGPipeline` and `AgentRAGWorkflow` alive inside the same MCP server session

## Documentation

Additional notes are in:

- `docs/mcp_integration_design.md`
- `docs/dev_log.md`

These files explain the MCP design, stdout/stderr handling, cache strategy, benchmark behavior, and known Windows + FastMCP cold-start issue.

## Design Goal

The project intentionally avoids heavy agent frameworks. The goal is a lightweight, runnable, and explainable POC that demonstrates how to build a local paper RAG system with incremental vector indexing, evaluation, Agent RAG, and MCP access.
