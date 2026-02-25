import streamlit as st
import os
import requests
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Engineer Verified Pro", layout="wide")

# Функция для кнопки копирования через JavaScript
def copy_to_clipboard(text, key):
    safe_text = json.dumps(text)
    js_code = f"""
    <button id="copy-btn-{key}" style="
        background-color: #0e1117; 
        color: #fafafa; 
        border: 1px solid #4d4d4d; 
        border-radius: 5px; 
        padding: 5px 10px; 
        cursor: pointer;
        font-size: 14px;
        margin-bottom: 10px;
        transition: 0.3s;">
        📋 Копировать ответ
    </button>
    <script>
    document.getElementById('copy-btn-{key}').onclick = function() {{
        const text = {safe_text};
        navigator.clipboard.writeText(text).then(() => {{
            this.innerText = '✅ Скопировано!';
            setTimeout(() => {{ this.innerText = '📋 Копировать ответ'; }}, 2000);
        }});
    }}
    </script>
    """
    components.html(js_code, height=45)

# --- 2. КОНФИГУРАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.0-flash" # Обновлено до актуальной стабильной версии
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={api_key}"

# --- 3. ОБРАБОТКА PDF ---
@st.cache_resource
def load_docs_engine():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    all_chunks = []
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    for f in files:
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {e}")
            
    return all_chunks

chunks = load_docs_engine()

def get_context(query, chunks):
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:8]
    
    raw_data = []
    context_text = ""
    for _, c in top_chunks:
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"SOURCE_{filename}_PAGE_{page_num}".replace(" ", "_")
        
        raw_data.append({
            "label": label, 
            "content": c.page_content, 
            "file": filename, 
            "page": page_num
        })
        context_text += f"\n--- ИСТОЧНИК_МЕТКА: {label} ---\n{c.page_content}\n"
        
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль")

if st.sidebar.button("Очистить историю чата"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        if m["role"] == "assistant":
            copy_to_clipboard(m["content"], f"hist_{i}")
            verified = m.get("verified_sources", [])
            if verified:
                with st.expander(f"✅ Подтверждающие выдержки"):
                    for src in verified:
                        st.success(f"**Источник: {src.get('file')}, стр. {src.get('page')}**")
                        st.text(src.get('content'))

# Поле ввода
if prompt := st.chat_input("Запросить технические данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализ документации и проверка источников..."):
            system_instruction = """
            Ты — промышленный ИИ-эксперт. Отвечай ТОЛЬКО по предоставленному контексту. 
            
            ФОРМАТ ССЫЛОК В КОНЦЕ:
            1. Создай заголовок '### Ссылки на документацию'.
            2. ПИШИ КАЖДУЮ ССЫЛКУ С НОВОЙ СТРОКИ ЧЕРЕЗ ДЕФИС (маркированный список).
            3. Шаблон одной строки: - <Название документа>, <Номер и название раздела>, стр. <номер>, <имя PDF-файла> (SOURCE_имяфайла_PAGE_номер)
            4. ВАЖНО: Метка в скобках (SOURCE_...) должна быть в конце каждой строки ссылки.
            5. Между основным текстом и ссылками сделай двойной отступ.
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                res_json = response.json()
                
                if 'error' in res_json:
                    clean_answer = f"❌ Ошибка API: {res_json['error'].get('message', 'Неизвестный сбой')}"
                    verified_sources = []
                elif 'candidates' in res_json and res_json['candidates']:
                    full_response = res_json['candidates'][0]['content']['parts'][0]['text']
                    verified_sources = [s for s in raw_candidates if s['label'] in full_response]
                    clean_answer = re.sub(r'\(SOURCE_.*?_PAGE_.*?\)', '', full_response)
                else:
                    # Случай, когда сработали фильтры безопасности Google
                    clean_answer = "⚠️ Ответ заблокирован фильтрами безопасности Google. Попробуйте перефразировать вопрос или проверьте содержимое документов."
                    verified_sources = []
                
            except Exception as e:
                clean_answer = f"❌ Ошибка связи: {str(e)}"
                verified_sources = []

        st.markdown(clean_answer)
        copy_to_clipboard(clean_answer, "new_msg")
        
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки"):
                for src in verified_sources:
                    st.success(f"**Файл: {src['file']}, Стр: {src['page']}**")
                    st.text(src['content'])
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clean_answer, 
            "verified_sources": verified_sources
        })
