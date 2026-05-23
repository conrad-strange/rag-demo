import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, Optional

from rag_service import (
    agent_security_answer,
    answer_security_question,
    benchmark_retrieval,
    get_index_status,
    list_knowledge_sources,
    retrieve_security_chunks,
    warmup_rag,
)


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "security-rag-mcp"
SERVER_VERSION = "0.1.0"


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_index_status": {
        "description": "Return local FAISS index, chunk, metadata, and document status for the security RAG knowledge base.",
        "handler": get_index_status,
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "list_knowledge_sources": {
        "description": "List indexed security knowledge sources and document categories.",
        "handler": list_knowledge_sources,
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "warmup_rag": {
        "description": "Preload the local embedding model and FAISS index so later retrieval calls reuse the cached RAG pipeline.",
        "handler": warmup_rag,
        "inputSchema": {
            "type": "object",
            "properties": {
                "use_rerank": {"type": "boolean", "description": "Also initialize reranker if enabled.", "default": False},
                "load_agent": {"type": "boolean", "description": "Also initialize the Agent RAG workflow wrapper.", "default": False},
            },
            "additionalProperties": False,
        },
    },
    "benchmark_retrieval": {
        "description": "Call retrieve_security_chunks twice in the same server session and report first/second call latency.",
        "handler": benchmark_retrieval,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User query or search phrase.", "default": "SQL injection prevention"},
                "category": {
                    "type": "string",
                    "description": "Document category filter.",
                    "default": "all",
                    "enum": ["all", "incident_response", "web_security", "llm_security"],
                },
                "top_k": {"type": "integer", "description": "Number of chunks to return.", "default": 2},
                "use_hybrid": {"type": "boolean", "description": "Use BM25 + vector hybrid retrieval.", "default": False},
                "use_rerank": {"type": "boolean", "description": "Use reranker after retrieval.", "default": False},
            },
            "additionalProperties": False,
        },
    },
    "retrieve_security_chunks": {
        "description": "Retrieve relevant security knowledge chunks without calling the LLM. Useful for grounding and debugging.",
        "handler": retrieve_security_chunks,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User query or search phrase."},
                "category": {
                    "type": "string",
                    "description": "Document category filter.",
                    "default": "all",
                    "enum": ["all", "incident_response", "web_security", "llm_security"],
                },
                "top_k": {"type": "integer", "description": "Number of chunks to return.", "default": 8},
                "use_hybrid": {"type": "boolean", "description": "Use BM25 + vector hybrid retrieval.", "default": False},
                "use_rerank": {"type": "boolean", "description": "Use reranker after retrieval.", "default": False},
                "include_text": {"type": "boolean", "description": "Include chunk text in the response.", "default": True},
                "max_chars": {"type": "integer", "description": "Maximum characters per returned chunk.", "default": 1200},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "answer_security_question": {
        "description": "Answer a security question with the normal RAG pipeline and return answer plus sources.",
        "handler": answer_security_question,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Security question to answer."},
                "category": {
                    "type": "string",
                    "description": "Document category filter.",
                    "default": "all",
                    "enum": ["all", "incident_response", "web_security", "llm_security"],
                },
                "vector_top_k": {"type": "integer", "description": "Vector retrieval candidate count.", "default": 8},
                "final_top_k": {"type": "integer", "description": "Final context chunk count.", "default": 3},
                "threshold": {"type": "number", "description": "Minimum vector score threshold.", "default": 0.4},
                "use_hybrid": {"type": "boolean", "description": "Use BM25 + vector hybrid retrieval.", "default": False},
                "use_rerank": {"type": "boolean", "description": "Use reranker after retrieval.", "default": True},
                "save_log": {"type": "boolean", "description": "Write local query log.", "default": True},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "agent_security_answer": {
        "description": "Answer a security question through Query Router + Tool Selection Agent RAG workflow.",
        "handler": agent_security_answer,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Security question to answer."},
                "category": {
                    "type": "string",
                    "description": "Document category filter.",
                    "default": "all",
                    "enum": ["all", "incident_response", "web_security", "llm_security"],
                },
                "vector_top_k": {"type": "integer", "description": "Vector retrieval candidate count.", "default": 8},
                "final_top_k": {"type": "integer", "description": "Final context chunk count.", "default": 3},
                "threshold": {"type": "number", "description": "Minimum vector score threshold.", "default": 0.4},
                "use_hybrid": {"type": "boolean", "description": "Use BM25 + vector hybrid retrieval.", "default": False},
                "use_rerank": {"type": "boolean", "description": "Use reranker after retrieval.", "default": True},
                "save_log": {"type": "boolean", "description": "Write local agent log.", "default": True},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _json_content(data: Any) -> list:
    return [
        {
            "type": "text",
            "text": json.dumps(data, ensure_ascii=False, indent=2),
        }
    ]


def _tool_specs() -> list:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
        }
        for name, spec in TOOL_REGISTRY.items()
    ]


def _success(request_id: Any, result: Any) -> Dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _error(request_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict:
    error: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def _call_tool(name: str, arguments: Optional[Dict]) -> Dict:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    handler: Callable = TOOL_REGISTRY[name]["handler"]
    arguments = arguments or {}
    # MCP stdio reserves stdout for protocol frames. Existing RAG code prints
    # model-loading/debug messages, so route those messages to stderr.
    with redirect_stdout(sys.stderr):
        result = handler(**arguments)
    return {
        "content": _json_content(result),
        "isError": False,
    }


def handle_request(request: Dict) -> Optional[Dict]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    try:
        if method == "initialize":
            return _success(
                request_id,
                {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            )

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return _success(request_id, {})

        if method == "tools/list":
            return _success(request_id, {"tools": _tool_specs()})

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            return _success(request_id, _call_tool(tool_name, arguments))

        return _error(request_id, -32601, f"Method not found: {method}")

    except Exception as exc:
        return _error(
            request_id,
            -32603,
            str(exc),
            {
                "traceback": traceback.format_exc(),
            },
        )


def read_message() -> Optional[Dict]:
    headers = {}

    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None

        line = line.decode("utf-8").strip()
        if not line:
            break

        key, _, value = line.partition(":")
        headers[key.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None

    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8"))


def write_message(message: Dict) -> None:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        request = read_message()
        if request is None:
            break

        response = handle_request(request)
        if response is not None:
            write_message(response)


if __name__ == "__main__":
    main()
