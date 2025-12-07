import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "db")

# --- 🚀 模型設定 ---
# 使用 3B 模型 (RTX 4050 順暢版)
LLM_MODEL = "qwen2.5:3b"

# Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-m3"

# 👇 補上這行！上下文視窗大小 (4050 建議 4096，以免爆顯存)
OLLAMA_NUM_CTX = 4096 

# 切片設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50