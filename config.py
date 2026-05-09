import os


# =========================
# 基础路径配置
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
LOG_DIR = os.path.join(BASE_DIR, "logs")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
INDEX_META_PATH = os.path.join(INDEX_DIR, "index_meta.json")

QUERY_LOG_PATH = os.path.join(LOG_DIR, "query_logs.jsonl")
EVAL_FILE_PATH = os.path.join(DATA_DIR, "eval_questions.csv")
EVAL_RESULT_PATH = os.path.join(LOG_DIR, "eval_results.csv")
EVAL_SUMMARY_PATH = os.path.join(LOG_DIR, "eval_summary.json")
THRESHOLD_COMPARE_PATH = os.path.join(LOG_DIR, "threshold_compare.csv")


# =========================
# 文档读取配置
# =========================

SUPPORTED_EXTENSIONS = [".txt", ".md", ".docx", ".pdf"]


# =========================
# 文本切分配置
# =========================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 60

# =========================
# Hybrid Search 配置
# =========================

USE_HYBRID_SEARCH = False
BM25_TOP_K = 8

# =========================
# 文档类别配置
# =========================

DOCUMENT_CATEGORIES = {
    "nist.sp.800-61r2.pdf": "incident_response",
    "owasp-top-10.pdf": "web_security",
    "OWASP-Top-10-for-LLMs-v2025.pdf": "llm_security"
}

CATEGORY_OPTIONS = [
    "all",
    "incident_response",
    "web_security",
    "llm_security"
]
# =========================
# Embedding 与向量检索配置
# =========================

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 第一阶段向量召回数量
VECTOR_TOP_K = 8

# 最终交给大模型的 chunk 数量
FINAL_TOP_K = 3

# 信息不足判断阈值
SIMILARITY_THRESHOLD = 0.4


# =========================
# Rerank 配置
# =========================

USE_RERANK = True

# 推荐中文/中英混合场景使用 bge-reranker-base
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"


# =========================
# LLM 配置
# =========================

LLM_PROVIDER = "deepseek"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
LLM_TEMPERATURE = 0.2


# =========================
# 运行时初始化目录
# =========================

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)