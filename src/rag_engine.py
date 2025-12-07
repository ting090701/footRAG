from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.config import DB_PATH, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_NUM_CTX

_CACHED_LLM = None
_CACHED_VECTORDB = None

def load_resources():
    global _CACHED_LLM, _CACHED_VECTORDB
    if _CACHED_LLM and _CACHED_VECTORDB:
        return _CACHED_LLM, _CACHED_VECTORDB

    print("⚙️ 初始化 RAG 引擎資源 (GPU)...")
    
    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

    _CACHED_VECTORDB = Chroma(persist_directory=DB_PATH, embedding_function=embedding_func)

    _CACHED_LLM = ChatOllama(
        model=LLM_MODEL, 
        temperature=0.1, # 溫度極低，減少亂發揮
        num_ctx=OLLAMA_NUM_CTX,
        num_gpu=1
    )
    
    return _CACHED_LLM, _CACHED_VECTORDB

def get_qa_chain():
    llm, vectordb = load_resources()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # (A) 改寫問題 Prompt (加強版)
    # 強制要求改寫後的語言必須跟原文一模一樣
    condense_prompt = PromptTemplate.from_template(
        """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        
        🔴 CRITICAL RULE:
        - You MUST maintain the **EXACT SAME LANGUAGE** as the user's input.
        - If user asks in English -> Output English.
        - If user asks in Chinese -> Output Traditional Chinese.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone Question:"""
    )

    # (B) 回答問題 Prompt (超級語言鎖版)
    qa_prompt = PromptTemplate.from_template(
        """You are 'FootAnalyzer', an expert AI assistant in physical therapy and podiatry.
        Use the following pieces of context to answer the user's question.

        ### 🔴 LANGUAGE RULES (MUST FOLLOW OR SYSTEM WILL FAIL):
        1. **DETECT** the language of the User's Question.
        2. **IF ENGLISH**: Answer entirely in **ENGLISH**. Do NOT use Chinese.
        3. **IF CHINESE**: Answer entirely in **TRADITIONAL CHINESE (繁體中文)**. 
           - **NEVER** use Simplified Chinese (简体中文).
        4. Match the user's language even if the Context is in a different language. Translate the knowledge if necessary.

        ### Guidelines:
        - Be professional, empathetic, and concise.
        - Use bullet points for clarity.
        - If you don't know the answer based on the context, say "I don't have information about that in my knowledge base" (in the user's language).

        ---
        Context:
        {context}

        User Question:
        {question}
        ---

        Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    
    return chain