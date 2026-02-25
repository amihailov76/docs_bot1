import streamlit as st
import os
import json
import re
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="MaxPatrol 10: Verified Engineer", layout="wide")

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

# --- 2. УМНАЯ ИНИЦИАЛИЗАЦИЯ (АВТОПОДБОР ИМЕНИ) ---
api_key = st.secrets.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

@st.cache_resource
def discover_model():
    """Автоматически находит правильное имя модели в текущем окружении"""
    try:
        # Получаем список всех моделей, доступных вашему ключу
        models = client.models.list()
        # Ищем 1.5 Flash, отдаем приоритет стабильным версиям без '2.'
        candidates = [m.name for m in models if "1.5-flash" in m.name.lower() and "2." not in m.name]
        
        if candidates:
            # Возвращаем имя (например, 'gemini-1.5-flash' или 'models/gemini-1.5-flash')
            return candidates[0]
        return "gemini-1.5-flash" # Резервный вариант
    except:
        return "gemini-1.5-flash"

STABLE_MODEL_ID = discover_model()

# --- 3. ОБРАБОТКА PDF ---
@st.cache_resource
def load_docs_engine():
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
    if st.button("🔄 Индексировать документы"):
        with st.spinner("Загрузка..."):
            st.session_state.chunks = load_docs_engine()
            st.success("База обновлена")
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.write(f"**Активное имя модели:**")
    st.code(STABLE_MODEL_ID)

def get_context(query, chunks):
    if not chunks: return [], ""
    q = query.lower()
    scored = []
    # Ваши приоритеты из сохраненной инструкции (adminguide, operatorguide, implementguide)
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

if prompt := st.chat_input("Ваш запрос..."):
    if not st.session_state.chunks: st.warning("Проиндексируйте базу.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            try:
                # Объединяем всё в один промпт для максимальной совместимости
                full_prompt = (
                    "Ты инженер техподдержки MaxPatrol 10. Твоя задача — отвечать на вопросы, "
                    "используя предоставленный ниже контекст. Отвечай кратко, по делу. "
                    "В конце обязательно укажи: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2...\n\n"
                    f"КОНТЕКСТ:\n{context}\n\n"
                    f"ВОПРОС: {prompt}"
                )
                
                response = client.models.generate_content(
                    model=STABLE_MODEL_ID,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                
                res = response.text
                clean = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', res).strip()
                ids = re.findall(r'ID_\d+', res)
                verified = [s for s in sources if s['label'] in ids] or sources[:1]
                links = "\n\n**Источники:**\n" + "\n".join([f"- {s['file']}, стр. {s['page']}" for s in verified])
                ans = clean + links
                st.markdown(ans)
                copy_to_clipboard(ans, "c")
                st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e: 
                st.error(f"Ошибка API: {e}")
                # Если все же ошибка, покажем список имен, чтобы понять, что не так
                try:
                    all_m = [m.name for m in client.models.list()]
                    st.write("Список доступных имен моделей:", all_m)
                except: pass
