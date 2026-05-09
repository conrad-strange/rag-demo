````markdown
# Security RAG Assistant v2

一个面向网络安全知识库的轻量级 RAG 问答与评估系统。

本项目从最小 RAG Demo 逐步扩展而来，当前已经支持真实英文安全 PDF 文档读取、文本切分、向量检索、Rerank、Metadata 分类过滤、PDF 表格解析、Hybrid Search、Streamlit 前端展示和基础评估实验。

项目定位是学习型与展示型 RAG 原型，不是企业级生产系统。

---

## 1. 环境搭建

建议使用 conda 创建独立环境：

```bash
conda create -n security-rag python=3.10 -y
conda activate security-rag
````

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 示例：

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

## 2. API Key 配置

在项目根目录新建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
```

注意：`.env` 文件不要上传到 GitHub。

---

## 3. 项目结构

```text
security-rag-v2/
├── app.py                  # Streamlit 前端
├── build_index.py           # 构建 FAISS 索引
├── config.py                # 参数配置
├── document_loader.py       # 文档读取
├── eval_rag.py              # 评估模块
├── hybrid_search.py         # BM25 / Hybrid Search
├── rag_pipeline.py          # RAG 核心流程
├── text_splitter.py         # 文本切分
├── requirements.txt
├── .env
│
├── data/
│   ├── nist.sp.800-61r2.pdf
│   ├── owasp-top-10.pdf
│   ├── OWASP-Top-10-for-LLMs-v2025.pdf
│   └── eval_questions.csv
│
├── index/
│   ├── faiss.index
│   ├── chunks.json
│   └── index_meta.json
│
├── logs/
│   ├── query_logs.jsonl
│   ├── eval_results.csv
│   ├── eval_summary.json
│   ├── threshold_compare.csv
│   └── rerank_compare.csv
│
└── notebooks/
```

---

## 4. 如何运行

### 4.1 构建知识库索引

```bash
python build_index.py
```

该步骤会读取 `data/` 下的文档，完成文本抽取、chunk 切分、embedding 向量化，并保存 FAISS 索引。

生成文件：

```text
index/faiss.index
index/chunks.json
index/index_meta.json
```

---

### 4.2 启动前端

```bash
streamlit run app.py
```

默认访问：

```text
http://localhost:8501
```

前端支持：

* 输入问题；
* 选择知识库范围；
* 开关 Rerank；
* 开关 Hybrid Search；
* 调整 Top-K；
* 调整 similarity threshold；
* 查看回答来源、chunk_id、vector_score、bm25_score、rerank_score。

---

### 4.3 运行评估

```bash
python eval_rag.py
```

评估结果会保存在：

```text
logs/eval_results.csv
logs/eval_summary.json
logs/threshold_compare.csv
logs/rerank_compare.csv
```

---

## 5. 当前已实现功能

当前系统已经实现：

* 支持 `.txt` / `.md` / `.docx` / 文本型 `.pdf` 文档读取；
* 使用 `pdfplumber` 对 PDF 文本和表格进行解析；
* 将 PDF 表格转换为 Markdown 文本后进入知识库；
* 使用段落优先策略进行 chunk 切分；
* 使用 sentence-transformers 生成 embedding；
* 使用 FAISS 建立本地向量索引；
* 支持 Metadata 分类过滤；
* 支持 BM25 + 向量检索的 Hybrid Search；
* 支持 Rerank 对候选 chunk 进行二次排序；
* 使用 DeepSeek API 生成回答；
* 使用 similarity threshold 判断知识库是否有足够信息；
* 对知识库外问题进行拒答；
* 保存 query log；
* 构建评估集并输出评估结果。

---

## 6. 当前知识库

当前使用三份真实英文安全 PDF：

| 文档                                | 主题                    | 分类                  |
| --------------------------------- | --------------------- | ------------------- |
| `nist.sp.800-61r2.pdf`            | 计算机安全事件处理指南           | `incident_response` |
| `owasp-top-10.pdf`                | OWASP Web Top 10      | `web_security`      |
| `OWASP-Top-10-for-LLMs-v2025.pdf` | OWASP LLM Top 10 2025 | `llm_security`      |

说明：NIST SP 800-61r2 当前不是最新版本，本项目仅将其作为历史安全文档测试样例。

---

## 7. 系统流程

```text
原始文档
→ 文档读取
→ PDF 文本与表格解析
→ 文本清洗
→ chunk 切分
→ embedding 向量化
→ FAISS 建索引
→ 用户提问
→ Metadata 分类过滤
→ 向量检索 / BM25 检索 / Hybrid Search
→ Rerank 重排序
→ Prompt 拼接
→ DeepSeek 生成回答
→ 展示答案与来源
→ 保存日志和评估结果
```

---

## 8. 评估结果

当前构建了 17 条测试问题：

* 13 条知识库内问题；
* 4 条知识库外问题。

当前一轮评估结果：

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

指标含义：

| 指标                     | 含义                 |
| ---------------------- | ------------------ |
| `source_hit_rate`      | Top-K 中包含预期来源文档的比例 |
| `top1_source_hit_rate` | Top1 就是预期来源文档的比例   |
| `avg_keyword_hit_rate` | 回答命中预期关键词的平均比例     |
| `refusal_accuracy`     | 知识库外问题被正确拒答的比例     |

从当前结果看，系统在真实英文 PDF 上具备基本可用的检索能力，并且通过 threshold 调整后，知识库外问题拒答准确率达到 100%。

---

## 9. Rerank 对比实验

为了验证 Rerank 是否改善检索排序，进行了 Rerank 开关对比。

```text
use_rerank=False
top1_source_hit_rate = 0.8462
avg_keyword_hit_rate = 0.5756

use_rerank=True
top1_source_hit_rate = 0.9231
avg_keyword_hit_rate = 0.5962
```

实验结果表明，在当前评估集上，开启 Rerank 后 Top1 来源命中率和平均关键词命中率均有提升。

需要注意的是，`rerank_score` 不是 0 到 1 的概率，也不是余弦相似度。它是 reranker 模型内部的相关性分数，可能为负数，主要用于比较候选 chunk 之间的相对相关性。

---

## 10. Threshold 对比实验

系统使用 `best_score` 判断知识库中是否存在足够相关的信息。

由于当前使用的是：

```python
normalize_embeddings=True
faiss.IndexFlatIP
```

因此 `vector_score` 可以近似理解为归一化向量之间的余弦相似度。

在实验中发现：

* threshold 较低时，系统更容易回答，但可能误答知识库外问题；
* threshold 较高时，系统更谨慎，拒答能力更强；
* 当前知识库和评估集下，`SIMILARITY_THRESHOLD = 0.40` 比 `0.35` 更稳定。

当前默认值：

```python
SIMILARITY_THRESHOLD = 0.40
```

---

## 11. 开发过程中遇到的问题与解决思路

### 11.1 Web Injection 与 Prompt Injection 混淆

问题：

```text
What are common web application Injection vulnerabilities and prevention methods?
```

在未加分类过滤时，系统容易检索到 OWASP LLM Top 10 中的 Prompt Injection，而不是 OWASP Web Top 10 中的 A03 Injection。

原因是：

```text
Injection 在 Web 安全中指 SQL Injection / Command Injection 等；
Injection 在 LLM 安全中可能指 Prompt Injection。
```

解决方法：

* 为每篇文档增加 `category`；
* 前端增加知识库范围选择；
* 支持 `incident_response` / `web_security` / `llm_security` 分类过滤；
* 修改检索逻辑：先扩大召回范围，再按 category 过滤，避免先全库 Top-K 后过滤导致正确 chunk 被漏掉。

这一问题说明，Metadata 不只是展示字段，它会直接影响真实检索效果。

---

### 11.2 PDF 表格内容回答不完整

在测试 NIST 文档时，部分表格类问题无法完整回答。

原因是：

* PDF 表格被普通文本抽取打散；
* 表格行列关系丢失；
* chunk 切分后表格上下文不完整；
* 普通问题可能优先命中正文，而不是表格 chunk。

解决方法：

* 引入 `pdfplumber`；
* 提取 PDF 表格；
* 将表格转为 Markdown 后写入知识库；
* 对表格类内容设计更明确的问题进行测试。

例如，普通问题：

```text
What are common sources of precursors and indicators during incident detection?
```

可能命中正文解释段落。

若要测试表格解析能力，更适合使用：

```text
Which sources are listed for precursors and indicators in the NIST incident detection table?
```

---

### 11.3 关键词命中率不能完全代表回答质量

最初关键词匹配区分大小写，导致部分正确回答被低估。

解决方法：

* 将关键词匹配改为大小写不敏感；
* 将关键词命中率作为辅助指标，而不是唯一判断标准。

当前仍存在的问题：

* 同义表达无法识别；
* 模型回答正确但未使用预设关键词时，命中率仍会偏低；
* 后续可以加入人工评分或 LLM-as-judge。

---

### 11.4 Rerank 分数为负数

在测试过程中发现 `rerank_score` 经常为负数。

这不是错误。Rerank 分数不是余弦相似度，也不是概率，而是模型内部相关性打分。应关注不同候选之间的相对大小，而不是简单判断正负。

例如：

```text
-2.05 > -2.29 > -5.90
```

表示第一个 chunk 在 reranker 看来最相关。

---

## 12. 当前不足

当前系统仍然是轻量级原型，主要不足包括：

1. PDF 解析能力仍有限
   对复杂表格、多栏排版、扫描版 PDF、图片文字支持不足。

2. chunk 策略仍较基础
   当前主要使用段落优先切分，尚未充分利用标题层级和 section metadata。

3. Hybrid Search 仍是基础实现
   当前只是 BM25 与向量检索候选合并，未做复杂分数融合。

4. 评估方式仍较简单
   关键词命中率无法充分评价语义正确性和 groundedness。

5. 缺少多轮对话能力
   当前主要支持单轮问答。

6. 缺少权限控制
   不适合直接用于企业敏感知识库。

---

## 13. 后续改进方向

后续可以继续改进：

* 增强 PDF 表格解析和版式恢复；
* 为 chunk 增加 section title 和 page number；
* 实现更合理的 Hybrid Search 分数融合；
* 增加 query rewrite；
* 增加人工评分字段和 failure_reason；
* 引入 groundedness / answer relevance 评估；
* 支持多轮对话；
* 增加文档级权限控制和日志脱敏。

