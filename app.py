import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant Ultimate", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Вход")
    pwd = st.text_input("Введите пароль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# --- 2. API КОНФИГУРАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")

# Список моделей для проверки
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
        return None, "В папке /docs нет PDF-файлов."
    
    try:
        all_docs = []
        for f in files:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            all_docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        return chunks, f"Документы загружены: {len(files)} шт. Чанков: {len(chunks)}."
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
st.title("🤖 Технический ассистент (Final Fix)")
st.info(status_msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Проверка всех доступных шлюзов Google API..."):
            context = get_context(prompt, chunks)
            res_text = ""
            error_log = []
            
            # Перебор версий API и названий моделей
            api_versions = ["v1", "v1beta"]
            
            found = False
            for version in api_versions:
                if found: break
                for model_name in MODELS_TO_TRY:
                    url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={api_key}"
                    
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": f"Ты тех-эксперт. Ответь детально, используя контекст.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС:\n{prompt}"
                            }]
                        }],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
                    }
                    
                    try:
                        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=15)
                        if response.status_code == 200:
                            res_json = response.json()
                            res_text = res_json['candidates'][0]['content']['parts'][0]['text']
                            found = True
                            break
                        else:
                            try:
                                err_msg = response.json().get('error', {}).get('message', 'Unknown Error')
                            except:
                                err_msg = response.text[:100]
                            error_log.append({
                                "Version": version,
                                "Model": model_name,
                                "Status": response.status_code,
                                "Message": err_msg
                            })
                    except Exception as e:
                        error_log.append({"Version": version, "Model": model_name, "Status": "Request Failed", "Message": str(e)})

            if not res_text:
                res_text = "### 🛑 Ошибка: Google API отклонил все запросы\n\n"
                res_text += "Ознакомьтесь с результатами диагностики ниже:\n"
                st.table(error_log)
                res_text += "\n**Что это значит?**\n- Если везде **404**: Модели не активированы для этого ключа.\n- Если везде **403**: Ваш IP (сервер Streamlit) заблокирован Google или ключ не имеет прав."
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
