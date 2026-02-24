import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant Fixed", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Вход")
    pwd = st.text_input("Введите пароль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверно")
    st.stop()

# --- 2. API КОНФИГУРАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
# Принудительно используем стабильный эндпоинт v1
API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

# --- 3. ОБРАБОТКА PDF ---
@st.cache_resource
def process_docs():
    docs_path = "./docs"
    if not os.path.exists(docs_path): os.makedirs(docs_path)
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files: return None, "Файлы не найдены."
    
    try:
        all_docs = []
        for f in files:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            all_docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        return chunks, f"Загружено чанков: {len(chunks)}"
    except Exception as e:
        return None, f"Ошибка PDF: {str(e)}"

chunks, status_msg = process_docs()

def get_context(query, chunks):
    if not chunks: return ""
    words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in words if w in c.page_content.lower())
        if score > 0: scored.append((score, c.page_content))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([c[1] for c in scored[:5]])

# --- 4. ИНТЕРФЕЙС ---
st.title("🤖 Технический ассистент")
st.caption(status_msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Связь с сервером Google..."):
            context = get_context(prompt, chunks)
            
            # Формируем JSON для прямого POST-запроса
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Ты тех-эксперт. Отвечай подробно по контексту.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС:\n{prompt}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            
            try:
                # Прямой HTTP запрос к стабильной версии API
                response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
                res_json = response.json()
                
                if response.status_code == 200:
                    res_text = res_json['candidates'][0]['content']['parts'][0]['text']
                else:
                    res_text = f"Ошибка API {response.status_code}: {res_json.get('error', {}).get('message', 'Unknown error')}"
            except Exception as e:
                res_text = f"Ошибка запроса: {str(e)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
