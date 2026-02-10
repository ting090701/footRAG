import uvicorn
import os
import re
import json

# 設定環境變數
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 引用您的 RAG 引擎
from src.rag_engine import get_qa_chain
from opencc import OpenCC

app = FastAPI(title="Moodle Local Brain")

cc = OpenCC('s2t')

# 允許跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    session_id: str = 'default'
    message: str

# 用來儲存對話歷史的字典
user_sessions = {}

# 預先載入大腦
print("🧠 正在啟動 RAG 引擎 (8B)...")
qa_chain = get_qa_chain()
print("✅ 引擎就緒！")

@app.post("/chat")
async def chat_endpoint(req: UserRequest):
    try:
        user_id = req.session_id
        
        # 1. 管理 Session 歷史紀錄
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        
        chat_history = []

        # 2. 呼叫 RAG
        # result 包含 'answer' 和 'source_documents' (需要在 rag_engine 設定 return_source_documents=True)
        result = qa_chain.invoke({
            "question": req.message,
            "chat_history": chat_history 
        })
        
        raw_answer = result.get('answer', '')
        source_docs = result.get('source_documents', [])

        # 3. 處理文字 (繁簡轉換 + 清洗 Markdown)
        answer_tc = cc.convert(raw_answer)
        clean_answer = re.sub(r'#+\s*', '', answer_tc) # 移除標題符號
        clean_answer = clean_answer.strip()

        # 🔥🔥🔥 新增：終端機 Debug 輸出 (讓您看 Chunk ID) 🔥🔥🔥
        print("\n" + "="*40)
        print(f"🆔 Session: {user_id}")
        print(f"🗣️ 用戶: {req.message}")
        print(f"🤖 AI: {clean_answer}")
        print("-" * 20)
        print("🔍 [DEBUG] 檢索到的 Chunk ID：")

        unique_ids = set()
        retrieved_chunks = []

        for doc in source_docs:
            # 優先從 metadata 抓取 ID
            c_id = doc.metadata.get("chunk_id") or doc.metadata.get("id")
            
            # 如果 metadata 沒有，嘗試從內容解析 (例如 [med#001])
            if not c_id:
                match = re.search(r'\[(med#\d+)\]', doc.page_content)
                if match:
                    c_id = match.group(1)
                else:
                    c_id = "Unknown"

            if c_id not in unique_ids:
                # 只印出前 30 個字供確認
                snippet = doc.page_content.replace('\n', '')[:30]
                print(f"   📄 ID: {c_id:<10} | 內容: {snippet}...")
                unique_ids.add(c_id)
                retrieved_chunks.append(c_id)
        
        print("="*40 + "\n")
        # 🔥🔥🔥 Debug 結束 🔥🔥🔥

        # 4. 更新歷史紀錄
        user_sessions[user_id].append((req.message, clean_answer))
        
        # 只留最後 10 句
        if len(user_sessions[user_id]) > 10:
             user_sessions[user_id] = user_sessions[user_id][-10:]

        # 5. 回傳結果
        # 除了 reply 外，也回傳 retrieved_chunk_ids 方便前端(如果有需要)顯示
        return {
            "reply": clean_answer,
            "retrieved_chunk_ids": retrieved_chunks,
            # "source_documents": [d.page_content for d in source_docs] # 若需要完整內容可解開註解
        }

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        # 在開發階段印出完整 traceback 比較好除錯
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 網頁託管 (Frontend_UI) ---
# 注意：確保這個檔案與 Frontend_UI 資料夾在同一層
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Frontend_UI")

@app.get("/")
async def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend_UI/index.html not found"}

# 掛載靜態檔案
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)