# Security RAG Assistant v2

Security RAG Assistant v2 is a local Retrieval-Augmented Generation demo for security knowledge QA. It builds a searchable knowledge base from real security documents, supports normal RAG and a lightweight Agent RAG workflow, and now exposes the same capabilities through an MCP server.

The project is designed as a learning and portfolio POC: small enough to understand, but complete enough to show document ingestion, indexing, retrieval, reranking, answer generation, evaluation, Streamlit UI, Agent-style routing, and MCP tool integration.

## Highlights

- Load PDF, TXT, Markdown, and DOCX security documents
- Split and clean documents into retrieval chunks
- Build a local FAISS vector index with SentenceTransformers embeddings
- Support optional BM25 + vector hybrid retrieval
- Support optional reranking with FlagEmbedding
- Generate grounded answers with DeepSeek Chat through an OpenAI-compatible client
- Provide a Streamlit UI with Normal RAG and Agent RAG modes
- Provide a Query Router + Tool Calling style Agent workflow
- Expose RAG and Agent RAG capabilities as MCP tools over stdio
- Include a local MCP client demo for tool listing, warmup, retrieval, and benchmark testing
- Record query logs, Agent logs, MCP stderr logs, and service performance logs

## Architecture

```text
User / Streamlit UI
-> app.py
-> RAGPipeline
-> FAISS / BM25 / reranker
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

MCP does not replace the RAG pipeline. It wraps the existing local RAG and Agent RAG capabilities behind a standard tool interface, so MCP hosts such as Claude Desktop, Cursor, Codex, or other clients can call the same project logic.

## Tech Stack

- Python 3.10
- Conda environment: `rag`
- Streamlit
- FAISS
- sentence-transformers
- rank-bm25
- FlagEmbedding
- MCP Python SDK / FastMCP
- DeepSeek API
- pandas / numpy
- pypdf / pdfplumber / python-docx

## Project Structure

```text
rag-demo-v2/
|-- app.py                         # Streamlit app: Normal RAG and Agent RAG
|-- agent_router.py                # Query Router + Tool Calling style Agent workflow
|-- build_index.py                 # Build FAISS index from files under data/
|-- config.py                      # Paths, retrieval, model, and LLM configuration
|-- document_loader.py             # PDF/TXT/Markdown/DOCX document loading
|-- eval_rag.py                    # RAG evaluation script
|-- hybrid_search.py               # BM25 and hybrid search helpers
|-- rag_pipline.py                 # Core RAG pipeline
|-- rag_service.py                 # Service layer used by MCP tools
|-- service_context.py             # Shared logging, timing, and env snapshots
|-- mcp_server_sdk.py              # Main FastMCP stdio server
|-- mcp_server.py                  # Manual stdio MCP server for protocol debugging
|-- mcp_client_demo.py             # Local MCP client demo
|-- requirements.txt
|-- .env.example
|-- data/                          # Source documents and eval questions
|-- docs/                          # MCP design notes and development log
|-- index/                         # Generated FAISS/chunks metadata; only .gitkeep is tracked
`-- logs/                          # Local generated logs; ignored by Git
```

## Quick Start

This project uses the existing Conda environment named `rag`.

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
```

The `.env` file is ignored by Git and should never be committed.

## Build the Index

Run this after the first clone or whenever files under `data/` change:

```powershell
conda activate rag
python build_index.py
```

Generated files:

```text
index/faiss.index
index/chunks.json
index/index_meta.json
```

These files are local generated artifacts and are ignored by Git.

## Run the Streamlit App

```powershell
conda activate rag
streamlit run app.py
```

After the embedding and rerank models are cached locally, offline mode can avoid slow Hugging Face network checks:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
conda run -n rag python -m streamlit run app.py
```

## App Modes

### Normal RAG

Normal RAG uses one fixed chain:

```text
User Query -> Retrieval -> Rerank -> Prompt -> DeepSeek -> Answer + Sources
```

It is suitable for direct security QA, such as:

```text
What is SQL injection?
How can an organization prepare for incident response?
What are common risks for LLM applications?
```

### Agent RAG

Agent RAG adds a lightweight Query Router and tool selection layer:

```text
User Query -> Query Router -> Tool Selection -> Retrieval/Prompt -> Answer + Sources
```

The router uses rule-based classification instead of an LLM classifier. Supported task types:

| Task type | Purpose |
| --- | --- |
| `fact_qa` | Direct factual security QA |
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
    "security-rag-mcp": {
      "command": "conda",
      "args": [
        "run",
        "-n",
        "rag",
        "python",
        "D:\\project\\rag\\rag-demo-v2\\mcp_server_sdk.py"
      ],
      "cwd": "D:\\project\\rag\\rag-demo-v2",
      "env": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false"
      }
    }
  }
}
```

For retrieval-heavy MCP demos on Windows, the local demo adds `--preload-retrieval-imports` when needed. This pre-imports `faiss`, `numpy`, and `sentence_transformers` before stdio serving to avoid a slow first import inside a tool request.

## Logs

| File | Purpose |
| --- | --- |
| `logs/query_logs.jsonl` | Normal RAG query logs |
| `logs/agent_log.jsonl` | Agent RAG debug logs |
| `logs/service_perf.log` | Service-layer timing and environment snapshots |
| `logs/mcp_client_demo_stderr.log` | MCP server stderr captured by the local demo client |

Important MCP note: stdio MCP uses stdout as the protocol channel. The MCP server redirects normal prints and model-loading messages to stderr so JSON-RPC frames are not polluted.

## Evaluation

Evaluation questions are stored in:

```text
data/eval_questions.csv
```

Run:

```powershell
conda run -n rag python eval_rag.py
```

Evaluation outputs are saved under `logs/` and treated as local generated artifacts.

## Useful Development Commands

Syntax check:

```powershell
conda run -n rag python -m py_compile service_context.py rag_service.py rag_pipline.py mcp_server_sdk.py mcp_server.py mcp_client_demo.py
```

Test the service layer directly without MCP:

```powershell
conda run -n rag python -c "from rag_service import benchmark_retrieval; import json; print(json.dumps(benchmark_retrieval(), ensure_ascii=False, indent=2))"
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

The project intentionally avoids heavy agent frameworks. The goal is a lightweight, runnable, and explainable POC that demonstrates how to add Agent and MCP capabilities to an existing local RAG application with minimal architectural overhead.
