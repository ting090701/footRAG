執行:
1.   .\.venv\Scripts\activate
2.   python server.py
3.   .\.venv\Scripts\activate
4.   .\ngrok http 8000

下載:
ollama pull qwen2.5:3b
python -m venv .venv
1.  .\.venv\Scripts\activate
2.  pip install -r requirements.txt
3.  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
4.  pip install fastapi uvicorn langchain langchain-community langchain-huggingface langchain-chroma chromadb pypdf pymupdf sentence-transformers neo4j langchain-experimental

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn
