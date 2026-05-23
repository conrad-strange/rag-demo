# MCP Integration Design

这份文档记录 `rag-demo-v2` 接入 MCP 的当前设计。定位是项目维护笔记，方便之后继续扩展、面试讲解和排查问题。

## 1. 原始 RAG 调用链

原始项目主要是一个本地 RAG 问答流程：

```text
User Query
-> Streamlit app.py
-> RAGPipeline
-> embedding query
-> FAISS vector search
-> optional BM25 hybrid search
-> optional reranker
-> prompt 拼接
-> DeepSeek/OpenAI-compatible LLM
-> answer + sources
```

核心对象是 `RAGPipeline`，它负责加载 embedding model、FAISS index、chunks、BM25 retriever、reranker 和 LLM client。

## 2. MCP 接入后的调用链

接入 MCP 后，RAG 能力被包装成一组工具，由 MCP client 通过 stdio 调用 MCP server：

```text
MCP Client
-> stdio MCP protocol
-> MCP Server
-> tool function
-> rag_service wrapper
-> cached RAGPipeline / cached AgentRAGWorkflow
-> retrieval / answer / agent answer
-> structured tool result
-> MCP Client
```

这里 MCP 不是替代 RAG，而是给 RAG 外面加了一层标准工具调用接口。后续如果接入 Claude Desktop、Cursor、Codex 或其他 MCP client，理论上都可以复用同一组工具。

## 3. 模块职责

### MCP Server

当前有两个 server：

- `mcp_server.py`：手写 stdio MCP server，主要用于理解 MCP 协议和调试。
- `mcp_server_sdk.py`：基于 Python MCP SDK / FastMCP 的正式版本，后续优先维护这个。

Server 的职责：

- 注册工具；
- 接收 MCP client 的 tool call；
- 调用 `rag_service.py` 中的服务函数；
- 返回结构化 JSON 结果；
- 保证 stdout 只输出 MCP 协议内容。

### MCP Client

`mcp_client_demo.py` 是本地测试 client。

职责：

- 启动 `mcp_server_sdk.py`；
- 初始化 MCP session；
- 列出工具；
- 调用轻量工具或真实检索工具；
- 打印 demo 结果。

### tools

MCP tools 是对外暴露的能力边界。它们不直接写复杂 RAG 逻辑，只做参数接收和服务函数调用。

### rag_service

`rag_service.py` 是 MCP 与 RAGPipeline 之间的 service 层。

职责：

- 统一工具函数接口；
- 做参数安全处理；
- 复用缓存后的 `RAGPipeline`；
- 复用缓存后的 `AgentRAGWorkflow`；
- 返回适合 MCP 输出的结构化结果；
- 提供 warmup 和 benchmark 能力。

### RAGPipeline

`rag_pipline.py` 仍然是底层 RAG 执行核心。

职责：

- 加载 embedding model；
- 加载 FAISS index；
- 加载 chunks；
- 初始化 BM25；
- 可选加载 reranker；
- 执行 vector / hybrid retrieval；
- 执行 rerank；
- 调 LLM 生成答案。

## 4. 当前暴露的工具列表

当前 SDK server 暴露这些工具：

| 工具名 | 用途 |
| --- | --- |
| `get_index_status` | 查看 FAISS、chunks、metadata 是否存在，以及文档数量、embedding 信息 |
| `list_knowledge_sources` | 列出当前知识库中的文档来源和分类 |
| `warmup_rag` | 预加载 embedding model、FAISS index、chunks、BM25，避免后续检索时冷启动 |
| `benchmark_retrieval` | 在同一个 server 会话内连续调用两次 retrieval，对比 first call / second call 耗时 |
| `retrieve_security_chunks` | 只做检索，不调用 LLM，适合调试 grounding 和 sources |
| `answer_security_question` | 普通 RAG 问答，返回 answer 和 sources |
| `agent_security_answer` | Agent RAG 问答，走 Query Router + tool selection 工作流 |

## 5. stdio MCP 的 stdout / stderr 注意事项

stdio MCP 最大的坑是：stdout 是协议通道，不能随便 print。

如果 server 在 stdout 打印普通日志、模型加载信息、进度条，就可能污染 MCP 协议，导致 client 解析失败。

当前处理方式：

- MCP server 的工具调用使用 `redirect_stdout(sys.stderr)`；
- 服务层性能日志写到 stderr；
- 同时写入 `logs/service_perf.log`；
- `mcp_client_demo.py` 把 server stderr 保存到 `logs/mcp_client_demo_stderr.log`；
- client 自己打印 demo 信息没问题，因为 client stdout 不是 server 协议通道。

## 6. mcp_client_demo.py 模式

### 默认轻量模式

```bash
python mcp_client_demo.py
```

默认只做：

- initialize MCP session；
- list tools；
- call `get_index_status`；
- call `list_knowledge_sources`。

这个模式不会加载 embedding model 和 FAISS index，适合快速演示 MCP 接入是否正常。

### 真实检索模式

```bash
python mcp_client_demo.py --with-retrieval
```

这个模式会：

- 启动 server 时预导入 retrieval 相关重依赖；
- 调用 `warmup_rag`；
- 调用 `retrieve_security_chunks`；
- 打印检索结果。

### Benchmark 模式

```bash
python mcp_client_demo.py --benchmark
```

这个模式会：

- warmup RAG；
- 在同一个 server 会话里连续调用两次 retrieval；
- 打印 first call 和 second call 耗时；
- 用来验证缓存是否生效。

## 7. 当前冷启动问题和后续优化方向

本轮遇到的主要问题是：`--with-retrieval` 在 Python MCP SDK / FastMCP server 下第一次真实检索会超时。

排查后发现：

- 不是 MCP 协议本身问题；
- 不是 FAISS index 或 chunks 缺失；
- 直接调用 `rag_service.benchmark_retrieval()` 可以正常完成；
- 真正的问题出现在 Windows + FastMCP tool request 中首次导入 `sentence_transformers -> transformers -> numpy` 非常慢，导致 240 秒 timeout。

当前处理：

- 默认轻量 demo 不预加载重依赖；
- 只有 `--with-retrieval`、`--warmup`、`--benchmark` 才给 server 加 `--preload-retrieval-imports`；
- server 启动阶段先预导入 `faiss`、`numpy`、`sentence_transformers`；
- 再通过 `warmup_rag` 加载 embedding model、FAISS、chunks；
- 后续 retrieval 直接复用缓存。

后续优化方向：

- 把 embedding model 路径改成更明确的本地目录，减少 HuggingFace cache 探测；
- 尝试更轻量 embedding 模型，减少首次加载时间；
- 将 `RAGPipeline` 拆得更细，例如 retrieval-only pipeline 不初始化 LLM client；
- 对 reranker 做单独 lazy loading，只有 `use_rerank=True` 才加载；
- 如果部署到 Linux，测试冷启动是否比 Windows 稳定；
- 后续接入真实 MCP host 时，保留当前 stderr 日志策略。

## 8. 本地测试方式

进入项目目录：

```bash
cd D:\project\rag\rag-demo-v2
```

先做语法检查：

```bash
python -m py_compile service_context.py rag_service.py rag_pipline.py mcp_server_sdk.py mcp_server.py mcp_client_demo.py
```

测试轻量 MCP demo：

```bash
python mcp_client_demo.py
```

测试完整检索链路：

```bash
python mcp_client_demo.py --with-retrieval
```

测试缓存效果：

```bash
python mcp_client_demo.py --benchmark
```

如果只想验证 service 层缓存，不经过 MCP：

```bash
python -c "from rag_service import benchmark_retrieval; import json; print(json.dumps(benchmark_retrieval(), ensure_ascii=False, indent=2))"
```

看性能日志：

```bash
type logs\service_perf.log
type logs\mcp_client_demo_stderr.log
```
