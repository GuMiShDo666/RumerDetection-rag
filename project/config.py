import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
RUMOR_DATA_DIR = os.path.join(_BASE_DIR, "data")
REFERENCE_DATA_DIR = os.path.join(RUMOR_DATA_DIR, "reference_data")
RUMOR_DATABASE_CSV = os.path.join(RUMOR_DATA_DIR, "rumor_database.csv")
RUMOR_DATABASE_MARKDOWN = os.path.join(MARKDOWN_DIR, "rumor_database.md")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "rumor_reference_articles"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = os.environ.get("DENSE_MODEL", "BAAI/bge-small-zh-v1.5")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "qwen").strip().lower()
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", os.environ.get("LLM_MODEL", "granite4.1:8b"))
LLM_MODEL = os.environ.get("LLM_MODEL", QWEN_MODEL if LLM_PROVIDER in {"qwen", "dashscope"} else OLLAMA_MODEL)
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_SEED = int(os.environ.get("LLM_SEED", "42"))
RAG_FORCE_RETRIEVAL_FALLBACK = os.environ.get("RAG_FORCE_RETRIEVAL_FALLBACK", "false").lower() == "true"
IMAGE_CAPTION_MODEL = os.environ.get("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
IMAGE_CAPTION_MAX_NEW_TOKENS = int(os.environ.get("IMAGE_CAPTION_MAX_NEW_TOKENS", "80"))
PADDLEOCR_LANG = os.environ.get("PADDLEOCR_LANG", "ch")

# --- Retrieval Configuration ---
RETRIEVAL_SCORE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 7
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Terminal Execution Logging ---
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", "100"))
MIN_PARENT_SIZE = int(os.environ.get("MIN_PARENT_SIZE", "2000"))
MAX_PARENT_SIZE = int(os.environ.get("MAX_PARENT_SIZE", "4000"))
INDEX_BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", "512"))
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
