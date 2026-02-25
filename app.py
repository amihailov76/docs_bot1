import streamlit as st
import os
import json
import re
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА ---
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

# --- 2. ПРЯМОЙ ВЫЗОВ GEMINI API (БЕЗ SDK) ---
API_KEY = st.secrets.get("GOOGLE_API_KEY")

def call_gemini_direct(prompt):
    """Прямой HTTP запрос к API v1 (самая стабильная ветка)"""
    # Используем v1 и 1.5-flash. Если и тут будет 404, значит Google 
    # в вашем регионе перевел всё на gemini-1.5-flash-latest
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except:
            return "Ошибка разбора ответа от ИИ."
    else:
        return f"Ошибка API {response.status_code}: {response.text}"

# --- 3. ЛОГИКА ДОКУМЕНТОВ ---
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

if "chunks" not in st.session_state: st.session_state.chunks = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    if st.button("🔄 Индексировать базу"):
        with st.spinner("Анализ PDF..."):
            st.session_state.chunks = load_docs()
            st.success("База готова!")
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("Движок: Gemini 1.5 Flash (Direct API v1)")

def get_context(query, chunks):
    if not chunks: return [], ""
    q = query.lower()
    scored = []
    priority = ['adminguide', 'operatorguide', 'implementguide']
    for c in chunks:
        txt = c.page_content.lower()
        fn = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for w in q.split() if len(w) > 3 and w in txt)
        if any(k in fn for k in priority): score *= 3.0
        else: score *= 0.6
        if score > 0: scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:7]
    raw = []; ctx = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        file = os.path.basename(c.metadata.get('source', ''))
        raw.append({"label": label, "file": file, "page": c.metadata.get('page', 0)+1})
        ctx += f"\n[{label}]\n{c.page_content}\n"
    return raw, ctx

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос по MP10..."):
    if not st.session_state.chunks: st.warning("Сначала проиндексируйте базу.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            
            full_prompt = (
                "Ты инженер техподдержки MaxPatrol 10. Отвечай кратко на основе контекста. "
                "В конце напиши: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2...\n\n"
                f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {prompt}"
            )
            
            with st.spinner("Думаю..."):
                res_text = call_gemini_direct(full_prompt)
            
            if "Ошибка API" in res_text:
                st.error(res_text)
            else:
                clean = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', res_text).strip()
                ids = re.findall(r'ID_\d+', res_text)
                verified = [s for s in sources if s['label'] in ids] or sources[:1]
                links = "\n\n**Источники:**\n" + "\n".join([f"- {s['file']}, стр. {s['page']}" for s in verified])
                ans = clean + links
                st.markdown(ans)
                copy_to_clipboard(ans, "c")
                st.session_state.messages.append({"role": "assistant", "content": ans})
