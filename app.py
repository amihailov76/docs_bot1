import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant Diagnostic", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Вход")
    pwd = st.text_input("Введите пароль доступа", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# --- 2. ПОДГОТОВКА API ---
api_key = st.secrets.get("GOOGLE_API_KEY")

# Список моделей для перебора (от самых новых к стабильным)
MODELS_TO_TRY = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro"
]

# --- 3. ОБРАБОТКА PDF ---
@st.cache_resource
def process_docs():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        return None, "В папке /docs на GitHub нет PDF-файлов."
    
    try:
        all_docs = []
        for f in files:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            all_docs.extend(loader.load())
        
        # Разбиваем на смысловые куски (чанкование)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(all_docs)
        return chunks, f"Документы считаны: {len(files)} шт. Создано фрагментов: {len(chunks)}."
    except Exception as e:
        return None, f"Ошибка при чтении PDF: {str(e)}"

chunks, status_msg = process_docs()

def get_context(query, chunks):
    if not chunks: return ""
    words = query.lower().split()
    scored = []
    for c in chunks:
        # Считаем совпадения слов для релевантности
        score = sum(1 for w in words if w in c
