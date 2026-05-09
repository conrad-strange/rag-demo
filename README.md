# RAG Security Demo

一个面向网络安全知识问答的最小化 RAG Demo。项目使用本地安全知识文档构建 FAISS 向量索引，通过 Streamlit 提供交互界面，并调用 DeepSeek Chat 生成基于检索结果的回答。

## 功能

- 从 `data/` 目录读取网络安全知识文本
- 使用 `sentence-transformers` 生成多语言文本向量
- 使用 FAISS 构建和查询本地向量索引
- 在 Streamlit 页面中输入问题并查看 Top-K 检索片段
- 使用 DeepSeek API 基于检索上下文生成回答
- 通过 `eval_rag.py` 对问答效果做简单评估

## 项目结构

```text
.
├── app.py                              # Streamlit RAG 问答应用
├── build_index.py                      # 构建 FAISS 索引
├── eval_rag.py                         # RAG 评估脚本
├── data/                               # 知识库文本和评估问题
├── index/                              # 已生成的 FAISS 索引和 chunks
├── outputs/                            # 本地评估输出，默认不提交
└── 01_minimal_rag_security_demo.ipynb.ipynb
```

## 环境要求

建议使用 Python 3.10 或更新版本。

安装依赖：

```bash
pip install streamlit python-dotenv sentence-transformers faiss-cpu numpy pandas openai
```

## 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

`.env` 已被 `.gitignore` 排除，不会上传到 GitHub。

## 构建索引

如果 `index/` 目录不存在，或者修改了 `data/` 下的知识文档，请重新构建索引：

```bash
python build_index.py
```

构建完成后会生成：

- `index/faiss.index`
- `index/chunks.json`

## 启动应用

```bash
streamlit run app.py
```

打开浏览器中的 Streamlit 地址后，输入网络安全相关问题即可查看回答、检索状态和 Top-K 参考片段。

## 运行评估

评估问题位于 `data/eval_questions.csv`：

```bash
python eval_rag.py
```

评估结果会输出到 `outputs/eval_results.csv`。`outputs/` 是本地运行产物，默认不提交到仓库。

## 注意事项

- 不要提交 `.env` 或任何 API Key。
- 如果更换 embedding 模型，需要重新执行 `python build_index.py`。
- 首次运行 `sentence-transformers` 可能会下载模型，请确保网络可用。
- 当前 Demo 面向教学和原型验证，不建议直接用于生产环境。
