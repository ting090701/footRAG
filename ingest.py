import os
import shutil
import json
import gc
import re
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document 
from src.config import DATA_PATH, DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# --- 資料清理函式 ---
def clean_text_content(text):
    """清理文件內容，去除頁碼、多餘空白與雜訊"""
    if not isinstance(text, str): return ""
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'- \d+ -', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def create_vector_db():
    print("📚 正在建立知識庫 (原子化 JSON + 切分 PDF)...")
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️ 請將檔案放入 {DATA_PATH}")
        return

    # 最終要寫入 DB 的所有文件
    final_docs = []
    
    # 用來同步寫入 JSONL 的列表 (給 BM25 用)
    jsonl_records = []

    # --- 1. 讀取 PDF (切分並標註 ID) ---
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if pdf_files:
        print(f"📄 發現 {len(pdf_files)} 個 PDF，讀取中...")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        try:
            raw_pdfs = loader.load()
            
            # 清理文字
            for doc in raw_pdfs:
                doc.page_content = clean_text_content(doc.page_content)
            
            # 切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
            pdf_chunks = text_splitter.split_documents(raw_pdfs)
            
            # 🔥 為每個 PDF Chunk 生成 ID
            for idx, chunk in enumerate(pdf_chunks):
                # 生成 ID：pdf_檔名雜湊_序號 (這裡簡化用 pdf_序號)
                chunk_id = f"pdf#{idx:04d}"
                
                # 更新 metadata
                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["source_type"] = "pdf"
                
                # ⭐️ 關鍵：將 ID 寫入內容，讓 Embedding 包含 ID 資訊
                # 原始內容保留在 metadata 以備不時之需
                original_text = chunk.page_content
                chunk.page_content = f"[{chunk_id}] {original_text}"
                
                final_docs.append(chunk)
                
                # 準備 JSONL 紀錄
                jsonl_records.append({
                    "id": chunk_id,
                    "text": original_text, # JSONL 存原始文字，server.py 會自己加 ID
                    "source": chunk.metadata.get("source", "pdf_doc")
                })
                
            print(f"   PDF 處理完成，共 {len(pdf_chunks)} 個片段")
            
        except Exception as e:
            print(f"❌ 讀取 PDF 錯誤: {e}")

    # --- 2. 讀取 JSON (原子化處理並標註 ID) ---
    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    for j_file in json_files:
        if j_file == "medical_docs_with_ids.jsonl": continue # 跳過自己產生的檔案

        print(f"📋 處理 JSON: {j_file} ...")
        try:
            path = os.path.join(DATA_PATH, j_file)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 統一轉成 List
            knowledge_list = []
            if isinstance(data, dict) and "medical_knowledge" in data:
                knowledge_list = data["medical_knowledge"]
            elif isinstance(data, list):
                knowledge_list = data
            else:
                knowledge_list = [json.dumps(data, ensure_ascii=False)]

            # 🔥 原子化並生成 ID
            for idx, content in enumerate(knowledge_list):
                # 如果 content 是字典 (例如原本就有 id 和 text)，嘗試提取
                if isinstance(content, dict):
                    text_content = content.get("text") or content.get("content") or str(content)
                    # 如果原檔有 ID 就用，沒有就生成
                    c_id = content.get("id") or content.get("chunk_id") or f"med#{idx:04d}"
                else:
                    text_content = str(content)
                    c_id = f"med#{idx:04d}"

                clean_content = clean_text_content(text_content)
                
                if clean_content:
                    # ⭐️ 關鍵：將 ID 寫入內容
                    doc_content = f"[{c_id}] {clean_content}"
                    
                    doc = Document(
                        page_content=doc_content,
                        metadata={
                            "chunk_id": c_id,
                            "source": j_file, 
                            "type": "atomic_knowledge"
                        }
                    )
                    final_docs.append(doc)
                    
                    # 準備 JSONL 紀錄
                    jsonl_records.append({
                        "id": c_id,
                        "text": clean_content,
                        "source": j_file
                    })
                    
        except Exception as e:
            print(f"❌ 讀取 JSON 失敗 ({j_file}): {e}")

    # 檢查總數
    if not final_docs:
        print("❌ 沒有有效資料可寫入資料庫！")
        return

    print(f"🧩 最終彙整: 共 {len(final_docs)} 個知識片段")

    # --- 3. 生成 JSONL 檔案 (給 BM25 用) ---
    jsonl_output_path = os.path.join(DATA_PATH, "medical_docs_with_ids.jsonl")
    print(f"💾 正在產生 BM25 用的 JSONL: {jsonl_output_path}")
    try:
        with open(jsonl_output_path, "w", encoding="utf-8") as f:
            for record in jsonl_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("✅ JSONL 檔案建立完成！")
    except Exception as e:
        print(f"❌ JSONL 建立失敗: {e}")

    # --- 4. 清理舊資料庫 ---
    gc.collect()
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("🗑️ 已清除舊向量資料庫")
        except:
            print("⚠️ 無法刪除舊資料庫，嘗試直接寫入...")

    # --- 5. 寫入 Chroma ---
    print(f"🚀 正在向量化並寫入 (Model: {EMBEDDING_MODEL})...")
    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True}
    )
    
    # 分批寫入
    batch_size = 5000
    for i in range(0, len(final_docs), batch_size):
        batch = final_docs[i:i+batch_size]
        Chroma.from_documents(
            documents=batch, 
            embedding=embedding_func, 
            persist_directory=DB_PATH
        )
        print(f"   已寫入批次 {i} ~ {i+len(batch)}")
        
    print(f"🎉 知識庫與索引檔建立完成！")

if __name__ == "__main__":
    create_vector_db()