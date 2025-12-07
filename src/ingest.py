import os
import shutil
import json
import gc  # 垃圾回收
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document 
from src.config import DATA_PATH, DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def create_vector_db():
    print("📚 正在建立知識庫 (支援 PDF 與 JSON)...")
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️ 請將檔案放入 {DATA_PATH}")
        return

    documents = []

    # --- 1. 讀取 PDF ---
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if pdf_files:
        print(f"📄 發現 {len(pdf_files)} 個 PDF，正在讀取...")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        try:
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ 讀取 PDF 時發生錯誤: {e}")

    # --- 2. 讀取 JSON ---
    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    if json_files:
        print(f"📋 發現 {len(json_files)} 個 JSON，正在讀取...")
        for j_file in json_files:
            try:
                path = os.path.join(DATA_PATH, j_file)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 轉字串
                text_content = json.dumps(data, ensure_ascii=False, indent=2)
                
                doc = Document(
                    page_content=text_content,
                    metadata={"source": j_file, "page": 0}
                )
                documents.append(doc)
            except Exception as e:
                print(f"❌ 讀取 JSON 失敗 ({j_file}): {e}")

    # 檢查有無資料
    if not documents:
        print("❌ data 資料夾中沒有可讀取的 PDF 或 JSON！")
        return

    # --- 3. 切分文字 (關鍵步驟：定義 texts 變數) ---
    print(f"✂️ 正在切分 {len(documents)} 份文件...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # 這行就是之前缺少的：把 documents 切成 texts
    texts = text_splitter.split_documents(documents)
    print(f"🧩 共切分為 {len(texts)} 個片段")

    # --- 4. 清理舊資料庫 (含垃圾回收) ---
    
    # 強制釋放記憶體，避免檔案被鎖定
    gc.collect()

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("🗑️ 已清除舊資料庫")
        except PermissionError:
            print("⚠️ 無法刪除舊資料庫 (可能正被佔用)，將嘗試直接覆蓋...")
        except Exception as e:
            print(f"⚠️ 清除資料庫時遇到小問題: {e}")

    # --- 5. 建立新資料庫 ---
    print(f"🚀 正在向量化並寫入資料庫 (使用 {EMBEDDING_MODEL})...")
    embedding_func = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    try:
        Chroma.from_documents(
            documents=texts,  # 這裡現在找得到 texts 了！
            embedding=embedding_func, 
            persist_directory=DB_PATH
        )
        print(f"✅ 知識庫建立完成！儲存於: {DB_PATH}")
    except Exception as e:
        print(f"❌ 建立資料庫失敗: {e}")

if __name__ == "__main__":
    create_vector_db()