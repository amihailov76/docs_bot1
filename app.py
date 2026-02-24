import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Documentation Assistant", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "123")

if "auth" not in st.session_state:
    st.title("🔐 Вход для инженеров")
    pwd = st.text_input("Введите пароль доступа", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        st.stop()

# --- 2. КОНФИГУРАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-flash" 
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_ID}:generateContent?key={api_key}"

# --- 3. ЗАГРУЗКА С МЕТАДАННЫМИ ---
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
                # Мы сохраняем источник в каждом чанке
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                all_chunks.extend(splitter.split_documents(pages))
        return all_chunks
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
        return None

chunks = load_docs()

def find_context_with_sources(query, chunks):
    if not chunks: return ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0:
            # Добавляем в текст чанка информацию об источнике
            source_info = f"\n[ИСТОЧНИК: {c.metadata.get('source', 'Неизвестен')}, Стр. {c.metadata.get('page', 0) + 1}]"
            scored.append((score, c.page_content + source_info))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n---\n\n".join([c[1] for c in scored[:8]])

# --- 4. ИНТЕРФЕЙС ---
st.title("🛡️ Инженерный Ассистент (Strict Mode)")
st.caption("Бот обязан цитировать документы и не имеет права на вымысел.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Запросить технические данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Поиск в официальной документации..."):
            context = find_context_with_sources(prompt, chunks)
            
            # ЖЕСТКИЙ ИНЖЕНЕРНЫЙ ПРОМПТ
            system_instruction = """
            ТЫ: Строгий технический эксперт.
            ТВОЯ ЗАДАЧА: Давать ответы ТОЛЬКО на основе предоставленного КОНТЕКСТА.
            
            ПРАВИЛА:
            1. Если в КОНТЕКСТЕ нет прямого ответа, напиши: "Информация в загруженных документах отсутствует".
            2. НЕ выдумывай параметры и цифры. 
            3. К каждому ключевому факту, числу или инструкции ДОБАВЛЯЙ ссылку на источник в формате: (Файл: [название], Стр: [номер]).
            4. В конце ответа выведи отдельный список "ИСПОЛЬЗОВАННЫЕ РАЗДЕЛЫ", где перечислишь все упомянутые документы и названия разделов (если они есть в тексте).
            5. Структурируй ответ технически грамотно: таблицы, списки, параметры.
            """
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"{system_instruction}\n\nКОНТЕКСТ ДЛЯ АНАЛИЗА:\n{context}\n\nВОПРОС ИНЖЕНЕРА:\n{prompt}"
                    }]
                }],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4000} 
            }
            # Установили temperature: 0.0 для максимальной точности (убираем творчество)
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                if response.status_code == 200:
                    answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    answer = "Ошибка доступа к API."
            except:
                answer = "Ошибка связи."
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
