# Security RAG Assistant v2

A lightweight Retrieval-Augmented Generation (RAG) assistant for security knowledge QA. The project builds a local knowledge base from real security documents and supports vector retrieval, BM25 hybrid search, reranking, DeepSeek answer generation, evaluation, Streamlit visualization, and a lightweight Agent Workflow extension.

This is a learning-oriented and portfolio-oriented AI application POC. It demonstrates an end-to-end path from document loading and indexing to RAG QA and an explainable Agent-style workflow.

## Features

- Load security documents from PDF, TXT, Markdown, and DOCX files
- Clean and split documents into chunks
- Generate embeddings with SentenceTransformers
- Store and search vectors with FAISS
- Support BM25 + vector hybrid search
- Support optional reranking with FlagEmbedding
- Generate grounded answers through DeepSeek Chat
- Provide a Streamlit web interface
- Provide basic RAG evaluation scripts
- Provide Query Router + Tool Calling style Agent RAG mode
- Record normal query logs and Agent debug logs

## Tech Stack

- Python
- Streamlit
- FAISS
- sentence-transformers
- rank-bm25
- FlagEmbedding
- DeepSeek API
- pandas / numpy
- pypdf / pdfplumber / python-docx

## Project Structure

```text
rag-demo-v2/
├── app.py                  # Streamlit app, supports Normal RAG and Agent RAG
├── agent_router.py          # Query Router + Tool Calling style Agent Workflow
├── build_index.py           # Build FAISS index
├── config.py                # Global configuration
├── document_loader.py       # Document loading
├── eval_rag.py              # RAG evaluation
├── hybrid_search.py         # BM25 and hybrid search
├── rag_pipline.py           # Core RAG pipeline
├── text_splitter.py         # Text splitting
├── requirements.txt
├── .env.example
├── data/                    # Source documents
├── index/                   # Local generated index files; GitHub keeps only .gitkeep
└── logs/                    # Local logs; ignored by Git
```

## Setup

Python 3.10 is recommended.

```bash
conda create -n security-rag python=3.10 -y
conda activate security-rag
pip install -r requirements.txt
```

If you already have a compatible environment, you can install the dependencies directly there.

## API Key

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

The `.env` file is ignored by Git and should not be uploaded to GitHub.

The DeepSeek API key is used by:

- Normal RAG mode answer generation
- Agent RAG mode tool answer generation
- `eval_rag.py` when running answer-generation evaluation

## Build the Index

Run this after the first clone or after changing documents under `data/`:

```bash
python build_index.py
```

Generated local files:

```text
index/faiss.index
index/chunks.json
index/index_meta.json
```

These generated index files are ignored by Git. Only `index/.gitkeep` is tracked so that the directory exists in the repository.

## Run the Streamlit App

```bash
streamlit run app.py
```

If the embedding and rerank models are already cached locally, offline mode can avoid slow Hugging Face network checks during startup:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
streamlit run app.py
```

On the current local machine, the existing `rag` environment can be used like this:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
D:\conda_env\envs\rag\python.exe -m streamlit run app.py
```

## App Modes

### Normal RAG Mode

Normal RAG mode uses one fixed chain for every question:

```text
User Query -> Retrieval -> Rerank -> Prompt -> DeepSeek -> Answer + Sources
```

It is suitable for standard security QA, for example:

```text
What is SQL injection?
```

### Agent RAG Mode

Agent RAG mode adds a lightweight Query Router and tool selection layer before the original RAG chain:

```text
User Query -> Query Router -> Tool Selection -> RAG Retrieval/Generation -> Answer + Sources
```

In the Streamlit UI, Agent mode also displays:

- `task_type`: the question type predicted by the router
- `tool_used`: the selected tool function
- `sources`: retrieved source chunks
- `status`: answer status
- `best_score`: highest vector similarity score

## Agent Workflow

The Agent logic is implemented in:

```text
agent_router.py
```

### Supported Query Types

The router uses simple rule matching instead of an LLM classifier:

- `fact_qa`: normal factual QA
- `table_qa`: table, field, column, or structured metadata questions
- `summary`: summary or overview questions
- `compare`: comparison or difference questions

### Tool Functions

The Agent Workflow exposes Tool Calling style functions:

- `search_docs(query)`: normal document retrieval QA
- `search_tables(query)`: prefer table-like or field-like chunks, then fall back to normal retrieval
- `summarize_docs(query)`: retrieve more chunks and use a summary prompt
- `compare_docs(query)`: retrieve more chunks and use a comparison prompt

These tools reuse the existing RAG pipeline:

- `vector_retrieve`
- `hybrid_retrieve`
- `rerank`
- `ask_llm`

This keeps the implementation small and avoids rebuilding the retrieval stack.

### Return Format

Agent mode returns a structured result:

```json
{
  "query": "Compare SQL injection and XSS.",
  "task_type": "compare",
  "tool_used": "compare_docs",
  "answer": "...",
  "sources": [
    {
      "source": "owasp-top-10.pdf",
      "chunk_id": 12,
      "category": "web_security"
    }
  ]
}
```

### Prompt Templates

Different task types use different lightweight prompt templates:

- `fact_qa`: answer only from retrieved context and avoid hallucination
- `table_qa`: prioritize table, field, or structured evidence
- `summary`: produce structured bullet summaries
- `compare`: compare with consistent dimensions and clear sections

## Example Questions

Try these in Agent RAG Mode:

```text
What is SQL injection?
Summarize the main security risks in this document.
Compare SQL injection and XSS.
What fields are included in the table?
```

## Logs

Normal RAG query logs:

```text
logs/query_logs.jsonl
```

Agent query logs:

```text
logs/agent_log.jsonl
```

Agent logs include:

- query
- task_type
- tool_used
- status
- top-k sources
- answer_length

## Evaluation

Run:

```bash
python eval_rag.py
```

Evaluation questions are stored in:

```text
data/eval_questions.csv
```

Evaluation outputs are saved under `logs/` and are treated as local generated artifacts.

## Performance Notes

If Streamlit startup is slow, the usual reasons are:

- loading the SentenceTransformer embedding model
- loading FAISS index and `chunks.json`
- initializing BM25 retriever
- loading the optional reranker model
- checking Hugging Face model files online

Useful optimizations:

- Set `USE_RERANK = False` in `config.py` for faster demos
- Use Hugging Face offline mode after models are cached
- Pre-download embedding and rerank models
- Keep `@st.cache_resource` for the RAG pipeline
- Disable Hybrid Search and Rerank when only demonstrating the Agent Workflow

## Why Add Agent Workflow

The original project was a fixed-chain RAG demo:

```text
Query -> Retrieval -> Prompt -> Answer
```

That is useful, but every user question is handled by the same retrieval and prompt strategy. In practice, different question types need different behavior:

- factual questions need concise grounded answers
- summary questions need more context and a summary prompt
- comparison questions need parallel dimensions and clear structure
- table or field questions should prefer structured chunks when possible

The Agent extension adds a small workflow layer:

```text
Query -> Router -> Tool -> Retrieval/Prompt -> Answer
```

This shows several AI Agent development skills without adding a heavy framework:

- Query Router design
- Tool Calling style interface design
- Agent Workflow orchestration
- Prompt Engineering for different task types
- Reuse of an existing RAG retrieval chain
- Explainable debug output in the UI
- Logging for later evaluation and optimization

The project intentionally does not use LangGraph, AutoGen, or CrewAI. The goal is a lightweight, runnable, and explainable POC that demonstrates how Agent capabilities can be added to an existing RAG system with minimal architectural changes.
