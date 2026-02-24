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
        score = sum(1 for w in words if w in c.page_content.lower())
        if score > 0:
            scored.append((score, c.page_content))
    
    # Сортируем и берем 5 самых подходящих кусков
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([c[1] for c in scored[:5]])

# --- 4. ИНТЕРФЕЙС ЧАТА ---
st.title("🤖 Технический ассистент (Diagnostic Mode)")
st.info(status_msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории чата
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Спросите что-нибудь из документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Диагностика каналов связи с Google..."):
            context = get_context(prompt, chunks)
            error_log = []
            res_text = ""
            
            # ЦИКЛ ПЕРЕБОРА МОДЕЛЕЙ
            for model_name in MODELS_TO_TRY:
                # Явно стучимся в версию v1 (стабильную)
                url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"Ты тех-эксперт. Используй только этот контекст для ответа:\n{context}\n\nВопрос: {prompt}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 2048
                    }
                }
                
                headers = {'Content-Type': 'application/json'}
                
                try:
                    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        res_text = res_json['candidates'][0]['content']['parts'][0]['text']
                        break # Нашли рабочую модель — выходим из цикла
                    else:
                        # Фиксируем причину отказа для каждой модели
                        try:
                            err_data = response.json()
                            msg = err_data.get('error', {}).get('message', 'Нет описания ошибки')
                        except:
                            msg = response.text[:100]
                        error_log.append(f"❌ {model_name}: Ошибка {response.status_code} ({msg})")
                
                except Exception as e:
                    error_log.append(f"⚠️ {model_name}: Ошибка запроса ({str(e)})")

            # Если ни одна модель не ответила
            if not res_text:
                res_text = "### 🛑 Не удалось получить ответ от Google API\n\n"
                res_text += "Я перебрал доступные модели, и вот результаты:\n"
                for log in error_log:
                    res_text += f"{log}\n"
                res_text += "\n---\n**Рекомендации:**\n1. Проверьте API-ключ в Settings -> Secrets.\n2. Убедитесь, что для вашего региона (IP сервера) разрешен Gemini.\n3. Попробуйте создать новый API-ключ в Google AI Studio."

        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
