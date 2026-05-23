# Development Log

这份日志记录 `rag-demo-v2` 接入 MCP 的开发过程。它不是论文式说明，而是给之后维护和复盘看的项目笔记。

## 1. 先做 service 层

最开始没有直接把 MCP tool 绑到 `RAGPipeline`，而是先抽了一层 `rag_service.py`。

这样做的原因：

- MCP tool 的输入输出需要比较稳定；
- 原始 RAG 代码里有些返回字段不适合直接暴露；
- 后续无论是手写 MCP server 还是 SDK server，都可以复用同一个 service 层；
- 也方便加缓存、日志和 benchmark。

service 层提供了这些函数：

- `get_index_status`
- `list_knowledge_sources`
- `retrieve_security_chunks`
- `answer_security_question`
- `agent_security_answer`
- `warmup_rag`
- `benchmark_retrieval`

## 2. 手写 MCP server

第一版做了 `mcp_server.py`，手写 stdio MCP server。

这一版的目的不是追求优雅，而是搞清楚 MCP 最基础的工作方式：

```text
stdin 读取 JSON-RPC request
-> 根据 method 分发
-> tools/list 返回工具列表
-> tools/call 调用本地函数
-> stdout 写回 JSON-RPC response
```

这个阶段最大的收获是理解了 stdio MCP 的硬规则：stdout 是协议通道，不能乱 print。

## 3. 安装并改用 Python MCP SDK

后面安装了 `mcp` 包，并新增了 `mcp_server_sdk.py`。

SDK 版本使用 `FastMCP` 注册工具，代码比手写 server 更接近真实项目：

```python
mcp = FastMCP("security-rag-mcp")

@mcp.tool(name="retrieve_security_chunks")
def tool_retrieve_security_chunks(...):
    return _call_service(...)
```

之后正式维护优先看 SDK 版本，手写版本保留为理解协议和排查问题的参考。

## 4. 修复 stdout 污染

RAG 代码里原本有不少 `print`，比如模型加载、FAISS 加载、reranker 加载。

在普通 Python 脚本里这没问题，但在 stdio MCP server 里会污染 stdout，导致 MCP client 解析协议失败。

修复方式：

- 在 MCP server 调用 service 函数时使用 `redirect_stdout(sys.stderr)`；
- 性能日志统一写 stderr；
- 额外写入 `logs/service_perf.log`；
- `mcp_client_demo.py` 把 server stderr 保存到 `logs/mcp_client_demo_stderr.log`。

## 5. 新增 SDK client demo

新增 `mcp_client_demo.py`，用于模拟真实 MCP client 调用 SDK server。

它做了几件事：

- 启动 `mcp_server_sdk.py`；
- 初始化 MCP session；
- `list_tools`；
- 调用 `get_index_status`；
- 调用 `list_knowledge_sources`；
- 可选调用真实 retrieval。

这个 demo 的价值是可以独立验证 MCP server 是否真的可用，而不是只看代码注册工具。

## 6. 默认轻量工具调用

默认运行：

```bash
python mcp_client_demo.py
```

只调用轻量工具，不加载 embedding model、FAISS index、reranker。

这样做是为了演示速度：

- MCP server 能快速启动；
- tool list 能快速返回；
- metadata/chunks 状态能快速查看；
- 不会每次打开 demo 都等模型加载。

## 7. --with-retrieval 真实检索超时问题

后来测试：

```bash
python mcp_client_demo.py --with-retrieval
```

发现真实检索路径会超时。client 已经保留了 `read_timeout_seconds=240`，但第一次 retrieval 仍然可能等不到 response。

一开始怀疑是：

- embedding model 冷启动太慢；
- FAISS index 加载慢；
- metadata/chunks 重复读取；
- reranker 重复初始化；
- MCP 协议被 stdout 污染。

逐步排查后发现，stdout 污染已经修掉，直接调用 service 层也能正常跑完。

## 8. 初步判断为模型 / FAISS 冷启动慢

第一轮判断重点放在模型和索引冷启动：

- embedding model 每次重复加载会很慢；
- FAISS index 每次重复加载没有必要；
- chunks / metadata 每次读文件也可以缓存；
- reranker 如果启用，加载成本更高；
- Agent workflow 如果每次创建，也会重复创建 pipeline。

所以决定做全局缓存和 lazy loading。

## 9. 全局缓存、lazy loading、分阶段耗时日志

新增 `service_context.py`，提供：

- `log_event`
- `timed_stage`
- `env_snapshot`

然后在 `rag_service.py` 中用 `@lru_cache` 缓存：

- `_cached_index_meta`
- `_cached_chunks`
- `get_rag_pipeline`
- `get_cached_agent_workflow`

这样 `retrieve_security_chunks`、`answer_security_question`、`agent_security_answer` 都通过同一个缓存入口获取 pipeline，避免重复初始化。

在 `rag_pipline.py` 中增加阶段日志：

- load config cost
- load embedding model cost
- load faiss index cost
- load chunks / metadata cost
- load reranker cost
- first query embedding cost
- faiss search cost
- rerank cost

同时保留：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

日志里会打印这些环境变量和模型名，方便判断是否走本地缓存。

## 10. 连续两次 benchmark

新增 `benchmark_retrieval`，在同一个 server 会话内连续调用两次 `retrieve_security_chunks`。

目的很简单：

如果缓存生效，第一次可能包含 warmup / 首次 encode 成本，第二次应该明显更快。

本轮验证结果中，经过 warmup 后：

```text
first_call_seconds 约 0.04s
second_call_seconds 约 0.02s
```

这说明同一个 MCP server 会话内已经复用了 pipeline，没有重复加载 embedding model、FAISS index 和 chunks。

## 11. Windows + FastMCP 的额外问题

排查中发现一个比较具体的问题：

在 Windows + FastMCP tool request 里，第一次导入 `sentence_transformers -> transformers -> numpy` 可能异常慢，甚至导致 240 秒超时。

直接在普通 Python 里调用 service 层可以完成，说明不是 RAG 逻辑本身坏了。

当前处理方式：

- 默认轻量 demo 不加载这些重依赖；
- `--with-retrieval`、`--warmup`、`--benchmark` 时，client 给 server 增加 `--preload-retrieval-imports`；
- server 启动时先预导入 `faiss`、`numpy`、`sentence_transformers`；
- 然后再进入 MCP stdio serving；
- 后续 tool request 里只做模型加载和检索，不再卡在首次导入。

## 12. 当前测试命令

语法检查：

```bash
python -m py_compile service_context.py rag_service.py rag_pipline.py mcp_server_sdk.py mcp_server.py mcp_client_demo.py
```

轻量 demo：

```bash
python mcp_client_demo.py
```

真实检索：

```bash
python mcp_client_demo.py --with-retrieval
```

缓存 benchmark：

```bash
python mcp_client_demo.py --benchmark
```

直接测 service 层：

```bash
python -c "from rag_service import benchmark_retrieval; import json; print(json.dumps(benchmark_retrieval(), ensure_ascii=False, indent=2))"
```

查看日志：

```bash
type logs\service_perf.log
type logs\mcp_client_demo_stderr.log
```

## 13. 后续计划

后续可以继续做：

- 更清晰的本地模型路径配置；
- retrieval-only pipeline，避免只检索时初始化 LLM client；
- reranker 独立 warmup；
- 接入真实 MCP host 测试，例如 Claude Desktop / Cursor；
- 增加 README 中的 MCP 使用说明；
- 如果部署到 Linux，重新测试冷启动和 vLLM / reranker 兼容性。
