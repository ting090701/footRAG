import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag_engine import get_qa_chain

app = FastAPI(title="Moodle Local Brain")

# 允許跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    message: str

# 預先載入大腦 (這樣第一次問才不會卡住)
print("🧠 正在啟動 RAG 引擎 (3B)...")
qa_chain = get_qa_chain()
print("✅ 引擎就緒！")

# --- API 接口 ---
@app.post("/chat")
async def chat_endpoint(req: UserRequest):
    try:
        # 呼叫 RAG
        result = qa_chain.invoke({"question": req.message})
        return {"reply": result['answer']}
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 網頁託管 (Frontend_UI) ---
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "Frontend_UI")

@app.get("/")
async def read_root():
    # 確保 index.html 存在
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend_UI/index.html not found"}

# 掛載靜態檔案
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)