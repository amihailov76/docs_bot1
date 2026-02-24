import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Source Verified", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "123")

if "auth" not in st.session_state:
    st.title("🔐 Авторизация системы")
    pwd = st.text_input("Пароль инженера", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        st.stop()

# --- 2. КОНФИГУРАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-flash" 
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_ID}:generateContent?key={api_key}"

# --- 3. ПОДГОТОВКА ДАННЫХ ---
@st.cache_resource
def load_docs_with_metadata():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    all_chunks = []
    for f in os.listdir(docs_path):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
    return all_chunks

chunks = load_docs_with_metadata()

def get_engineered_context(query, chunks):
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:6]
    
    context_text = ""
    raw_data_for_ui = []
    
    for score, c in top_chunks:
        source_name = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        header = f"ДОКУМЕНТ: {source_name} | СТРАНИЦА: {page_num}"
        
        context_text += f"\n--- {header} ---\n{c.page_content}\n"
        raw_data_for_ui.append({"header": header, "content": c.page_content})
        
    return raw_data_for_ui, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль документации")
st.info("Режим: Верификация источников включена. Температура: 0.0 (Strict)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "raw_sources" in m:
            with st.expander("🔍 Показать исходные выдержки из PDF"):
                for src in m["raw_sources"]:
                    st.caption(f"**{src['header']}**")
                    st.code(src['content'], language=None)

if prompt := st.chat_input("Введите технический запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_sources, context_for_ai = get_engineered_context(prompt, chunks)
        
        with st.spinner("Проверка регламентов..."):
            system_instruction = """
            Ты — промышленный ИИ-ассистент. Твоя задача — извлекать точные данные из тех-документации.
            ПРАВИЛА ОТВЕТА:
            1. Используй ТОЛЬКО предоставленный текст.
            2. После КАЖДОГО утверждения или цифры ставь ссылку: **[Файл, Стр]**.
            3. Если информации нет, прямо ответь: "Данные не обнаружены в базе".
            4. Оформляй списки через дефис, важные параметры выделяй жирным.
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
            except:
                answer = "❌ Ошибка обработки запроса."

        st.markdown(answer)
        
        # Добавляем блок верификации
        with st.expander("🔍 Показать исходные выдержки из PDF"):
            for src in raw_sources:
                st.caption(f"**{src['header']}**")
                st.code(src['content'], language=None)
        
        # Сохраняем в историю вместе с источниками
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "raw_sources": raw_sources
        })
