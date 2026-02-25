import streamlit as st
import os
import json
import re
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Verified Pro", layout="wide")

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
    </script>
    """
    components.html(js_code, height=45)

# --- 2. ИНИЦИАЛИЗАЦИЯ БЕЗ 404 ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model_instance():
    variants = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-flash-latest"]
    for v in variants:
        try:
            m = genai.GenerativeModel(model_name=v)
            m.generate_content("test", generation_config={"max_output_tokens": 1})
            return m, v
        except Exception as e:
            if "429" in str(e): return genai.GenerativeModel(model_name=v), v
            continue
    return genai.GenerativeModel(model_name="gemini-1.5-flash"), "gemini-1.5-flash"

model, ACTIVE_MODEL_NAME = get_working_model_instance()

# --- 3. ЛОГИКА PDF ---
def load_docs_engine():
    docs_path = "./docs"
    if not os.path.exists(docs_path): return []
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    all_chunks = []
    progress = st.progress(0)
    for i, f in enumerate(files):
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except: continue
        progress.progress((i + 1) / len(files))
    return all_chunks

if "chunks" not in st.session_state: st.session_state.chunks = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    if st.button("🔄 Обновить базу PDF"):
        st.session_state.chunks = load_docs_engine()
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.success(f"Движок: `{ACTIVE_MODEL_NAME}`")

def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    scored = []
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    for c in chunks:
        content_low = c.page_content.lower()
        filename = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for w in query_low.split() if len(w) > 3 and w in content_low)
        if any(k in filename for k in priority_keywords): score *= 3.0
        else: score *= 0.6
        if score > 0: scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:7]
    raw_data = []; context_text = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        file = os.path.basename(c.metadata.get('source', ''))
        page = c.metadata.get('page', 0) + 1
        raw_data.append({"label": label, "content": c.page_content, "file": file, "page": page})
        context_text += f"\n[{label}]\n{c.page_content}\n"
    return raw_data, context_text

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    if not st.session_state.chunks: st.warning("Обновите базу PDF слева.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            sys_instr = "Ты инженер MP10. Отвечай кратко по контексту. В конце напиши: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1..."
            try:
                response = model.generate_content(f"{sys_instr}\n\nКонтекст:\n{context}\n\nВопрос: {prompt}")
                clean_text = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', response.text).strip()
                used_ids = re.findall(r'ID_\d+', response.text)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                links = "\n\n**Ссылки:**\n" + "\n".join([f"- {s['file']}, стр. {s['page']}" for s in verified])
                full_res = clean_text + links
                st.markdown(full_res)
                copy_to_clipboard(full_res, "last")
                st.session_state.messages.append({"role": "assistant", "content": full_res, "sources": verified})
            except Exception as e: st.error(f"Ошибка API: {e}")
