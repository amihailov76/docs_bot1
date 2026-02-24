import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant 2026", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "123")

if "auth" not in st.session_state:
    st.title("🔐 Вход")
    pwd = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        st.stop()

# --- 2. КОНФИГУРАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
# Используем модель, которую выдала диагностика
MODEL_ID = "gemini-2.5-flash" 
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_ID}:generateContent?key={api_key}"

# --- 3. ОБРАБОТКА PDF ---
@st.cache_resource
def load_docs():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    all_chunks = []
    try:
        for f in os.listdir(docs_path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_path, f))
                pages = loader.load()
                # 2.5 Flash отлично работает с длинным контекстом, чанки по 2000 знаков идеальны
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
                all_chunks.extend(splitter.split_documents(pages))
        return all_chunks
    except Exception as e:
        st.error(f"Ошибка загрузки PDF: {e}")
        return None

chunks = load_docs()

def find_context(query, chunks):
    if not chunks: return ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0: scored.append((score, c.page_content))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Передаем 7 чанков (Gemini 2.5 легко это переварит)
    return "\n\n".join([c[1] for c in scored[:7]])

# --- 4. ИНТЕРФЕЙС ---
st.title("🤖 Технический ассистент (Gemini 2.5 Flash)")
st.caption("База знаний активна и использует актуальную модель 2026 года.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос по документам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую документацию..."):
            context = find_context(prompt, chunks)
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Ты тех-эксперт. Используй контекст для ПОДРОБНОГО ответа. Если в тексте нет ответа, ответь сам, но уточни это.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС:\n{prompt}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 4000
                }
            }
            
            try:
                response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
                if response.status_code == 200:
                    answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    err = response.json().get('error', {}).get('message', 'Unknown Error')
                    answer = f"Ошибка API ({response.status_code}): {err}"
            except Exception as e:
                answer = f"Ошибка связи: {str(e)}"
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
