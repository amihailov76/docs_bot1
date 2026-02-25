import streamlit as st
import os
import json
import re
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="MP10 Verified Engineer", layout="wide")

def copy_to_clipboard(text, key):
    safe_text = json.dumps(text)
    js_code = f"""
    <button id="copy-btn-{key}" style="background-color: #0e1117; color: #fafafa; border: 1px solid #4d4d4d; border-radius: 5px; padding: 5px 10px; cursor: pointer; font-size: 14px; margin-bottom: 10px;">📋 Копировать ответ</button>
    <script>
    document.getElementById('copy-btn-{key}').onclick = function() {{
        navigator.clipboard.writeText({safe_text}).then(() => {{
            this.innerText = '✅ Скопировано!';
            setTimeout(() => {{ this.innerText = '📋 Копировать ответ'; }}, 2000);
        }});
    }}
    </script>"""
    components.html(js_code, height=45)

# --- 2. ПАРАМЕТРЫ МОДЕЛИ (ФИНАЛЬНЫЙ ФИКС ФОРМАТА) ---
API_KEY = st.secrets.get("GOOGLE_API_KEY")
# Берем имя из вашего списка. Если оно с "models/", убираем префикс для URL
RAW_MODEL_NAME = "models/gemini-flash-latest"
CLEAN_MODEL_NAME = RAW_MODEL_NAME.replace("models/", "")

def call_gemini(prompt):
    # Формируем URL. В v1beta/v1 адрес должен быть: /models/{name}:generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CLEAN_MODEL_NAME}:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    
    # ВАЖНО: В payload НЕ ДОЛЖНО быть поля "model", 
    # так как имя уже есть в URL. Только "contents".
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Ошибка API {res.status_code}: {res.text}"
    except Exception as e:
        return f"Ошибка соединения: {str(e)}"

# --- 3. ОБРАБОТКА ДОКУМЕНТОВ ---
@st.cache_resource
def load_docs():
    docs_path = "./docs"
    if not os.path.exists(docs_path): return []
    all_chunks = []
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    for f in files:
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except: continue
    return all_chunks

def get_context(query, chunks):
    if not chunks: return [], ""
    q = query.lower()
    scored = []
    # Поиск по вашим приоритетам: adminguide, operatorguide, implementguide
    priority_keys = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        txt = c.page_content.lower()
        fn = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for word in q.split() if len(word) > 3 and word in txt)
        
        if any(key in fn for key in priority_keys):
            score *= 3.0
        else:
            score *= 0.5
        if score > 0: scored.append((score, c))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:6]
    
    raw = []; ctx_text = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        fname = os.path.basename(c.metadata.get('source', ''))
        page = c.metadata.get('page', 0) + 1
        raw.append({"label": label, "file": fname, "page": page})
        ctx_text += f"\n[{label}] (Файл: {fname}, Стр: {page})\n{c.page_content}\n"
    return raw, ctx_text

# --- 4. ИНТЕРФЕЙС И ЧАТ ---
st.title("🏗️ MP10: Verified Engineer")

if "messages" not in st.session_state: st.session_state.messages = []
if "chunks" not in st.session_state: st.session_state.chunks = None

with st.sidebar:
    st.header("Управление")
    if st.button("🔄 Проиндексировать PDF"):
        with st.spinner("Анализ документов..."):
            st.session_state.chunks = load_docs()
            st.success(f"Готово! Чанков: {len(st.session_state.chunks)}")
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.success(f"Движок: `Gemini 1.5 Flash` (Высокий лимит)")

# Чат
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Задайте вопрос по MaxPatrol 10..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.chunks:
        with st.chat_message("assistant"): st.warning("Сначала проиндексируйте PDF в боковом меню.")
    else:
        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            full_prompt = (
                "Ты эксперт по MaxPatrol 10. Отвечай только по контексту. "
                "В конце напиши: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2...\n\n"
                f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {prompt}"
            )
            with st.spinner("Думаю..."):
                response = call_gemini(full_prompt)
            
            if "Ошибка API 429" in response:
                st.error("Лимит исчерпан. Подождите 60 секунд.")
            else:
                clean_ans = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', response).strip()
                used_ids = re.findall(r'ID_\d+', response)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                source_links = "\n\n**Источники:**\n" + "\n".join([f"- {s['file']}, стр. {s['page']}" for s in verified])
                final_text = clean_ans + source_links
                st.markdown(final_text)
                copy_to_clipboard(final_text, f"ans_{len(st.session_state.messages)}")
                st.session_state.messages.append({"role": "assistant", "content": final_text})
