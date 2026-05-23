import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


BASE_DIR = Path(__file__).resolve().parent
SERVER_PATH = BASE_DIR / "mcp_server_sdk.py"
ERRLOG_PATH = BASE_DIR / "logs" / "mcp_client_demo_stderr.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo client for the local Security RAG MCP SDK server.")
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="Run a real retrieval call after warming the local RAG pipeline.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up embedding model + FAISS index without running a retrieval query.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Call retrieve_security_chunks twice in one server session and print first/second timings.",
    )
    return parser.parse_args()


def _text_content_to_json(tool_result):
    if not tool_result.content:
        return None
    first = tool_result.content[0]
    text = getattr(first, "text", "")
    return json.loads(text)


async def run_demo() -> None:
    args = parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    server_args = [str(SERVER_PATH)]
    if args.with_retrieval or args.warmup or args.benchmark:
        server_args.append("--preload-retrieval-imports")

    server = StdioServerParameters(
        command=sys.executable,
        args=server_args,
        cwd=str(BASE_DIR),
        env=env,
        encoding="utf-8",
        encoding_error_handler="replace",
    )

    ERRLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERRLOG_PATH, "w", encoding="utf-8") as errlog:
        async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=240),
            ) as session:
                await session.initialize()

                tools = await session.list_tools()
                print("Available tools:", flush=True)
                for tool in tools.tools:
                    print(f"- {tool.name}: {tool.description}", flush=True)

                status_result = await session.call_tool("get_index_status", {})
                status = _text_content_to_json(status_result)
                print("\nIndex status:", flush=True)
                print(json.dumps(status, ensure_ascii=False, indent=2)[:800], flush=True)

                sources_result = await session.call_tool("list_knowledge_sources", {})
                sources = _text_content_to_json(sources_result)
                print("\nKnowledge sources:", flush=True)
                print(json.dumps(sources, ensure_ascii=False, indent=2)[:800], flush=True)

                if args.with_retrieval or args.warmup or args.benchmark:
                    warmup_result = await session.call_tool(
                        "warmup_rag",
                        {
                            "use_rerank": False,
                            "load_agent": False,
                        },
                    )
                    warmed = _text_content_to_json(warmup_result)
                    print("\nWarmup:", flush=True)
                    print(json.dumps(warmed, ensure_ascii=False, indent=2)[:1000], flush=True)

                if args.benchmark:
                    benchmark_result = await session.call_tool(
                        "benchmark_retrieval",
                        {
                            "query": "SQL injection prevention",
                            "top_k": 2,
                            "use_rerank": False,
                        },
                    )
                    benchmark = _text_content_to_json(benchmark_result)
                    print("\nBenchmark:", flush=True)
                    print(json.dumps(benchmark, ensure_ascii=False, indent=2), flush=True)

                if args.with_retrieval:
                    retrieve_result = await session.call_tool(
                        "retrieve_security_chunks",
                        {
                            "query": "SQL injection prevention",
                            "top_k": 2,
                            "use_rerank": False,
                            "include_text": False,
                            "max_chars": 220,
                        },
                    )
                    retrieved = _text_content_to_json(retrieve_result)
                    print("\nRetrieved chunks:", flush=True)
                    print(json.dumps(retrieved, ensure_ascii=False, indent=2)[:1200], flush=True)
                elif not args.warmup and not args.benchmark:
                    print(
                        "\nSkipped retrieval demo. Pass --with-retrieval to load the local embedding model and FAISS index.",
                        flush=True,
                    )


if __name__ == "__main__":
    anyio.run(run_demo)
