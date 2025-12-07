import fitz  # PyMuPDF
import os
from src.config import DATA_PATH, IMAGE_DIR

def extract_images():
    print("🖼️ 正在提取 PDF 圖片...")
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            doc = fitz.open(os.path.join(DATA_PATH, pdf_file))
            file_name = os.path.splitext(pdf_file)[0]
            
            for i, page in enumerate(doc):
                # 2倍放大以確保清晰
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_path = os.path.join(IMAGE_DIR, f"{file_name}_{i}.png")
                if not os.path.exists(img_path):
                    pix.save(img_path)
            print(f"✅ {pdf_file} 圖片提取完成")
        except Exception as e:
            print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    extract_images()