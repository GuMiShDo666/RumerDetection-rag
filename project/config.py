import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
RUMOR_DATA_DIR = os.path.join(_BASE_DIR, "data")
TRAIN_CSV = os.path.join(RUMOR_DATA_DIR, "train.csv")
VALID_CSV = os.path.join(RUMOR_DATA_DIR, "valid.csv")
TEST_CSV = os.path.join(RUMOR_DATA_DIR, "test.csv")
RUMOR_DATABASE_CSV = os.path.join(RUMOR_DATA_DIR, "rumor_database.csv")
RUMOR_DATABASE_MARKDOWN = os.path.join(MARKDOWN_DIR, "rumor_database.md")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "rumor_detection_cases"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "granite4.1:8b"
LLM_TEMPERATURE = 0
LLM_SEED = 42
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
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
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
