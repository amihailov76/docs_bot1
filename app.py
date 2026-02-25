import streamlit as st
import os
import json
import re
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА ИНТЕРФЕЙСА ---
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

# --- 2. ИНИЦИАЛИЗАЦИЯ SDK (GOOGLE-GENAI) ---
api_key = st.secrets.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Фиксируем Gemini 1.5 Flash (1500 запросов в день)
STABLE_MODEL_ID = "gemini-1.5-flash"

# --- 3. ОБРАБОТКА PDF С ПРИОРИТЕТАМИ ---
@st.cache_resource
def load_docs_engine():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        return []
    
    all_chunks = []
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    for f in files:
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.sidebar.error(f"Ошибка в {f}: {e}")
            continue
    return all_chunks

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. ЛОГИКА ПОИСКА (RAG) ---
def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    scored = []
    
    # Приоритеты из вашей инструкции
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        fname = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for w in query_low.split() if len(w) > 3 and w in content_low)
        
        if any(k in fname for k in priority_keywords):
            score *= 3.0
        else:
            score *= 0.6
            
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:7]
    
    raw_data = []
    context_text = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        file = os.path.basename(c.metadata.get('source', ''))
        page = c.metadata.get('page', 0) + 1
        raw_data.append({"label": label, "content": c.page_content, "file": file, "page": page})
        context_text += f"\n--- ИСТОЧНИК {label} ---\n{c.page_content}\n"
    return raw_data, context_text

# --- 5. ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    if st.button("🔄 Индексировать документы"):
        with st.spinner("Чтение PDF..."):
            st.session_state.chunks = load_docs_engine()
            st.success("База знаний обновлена")
    
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.success(f"Движок: `{STABLE_MODEL_ID}`")

# Чат
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ваш технический запрос..."):
    if not st.session_state.chunks:
        st.warning("Сначала проиндексируйте PDF в левой панели.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            
            try:
                config = types.GenerateContentConfig(
                    system_instruction=(
                        "Ты ведущий инженер техподдержки MaxPatrol 10. Отвечай только по контексту. "
                        "В конце укажи: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2"
                    ),
                    temperature=0.1
                )
                
                response = client.models.generate_content(
                    model=STABLE_MODEL_ID,
                    contents=f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {prompt}",
                    config=config
                )
                
                res_text = response.text
                clean_ans = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', res_text).strip()
                used_ids = re.findall(r'ID_\d+', res_text)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                
                links = "\n\n**Источники:**\n" + "\n".join([
                    f"- {s['file'].replace('_',' ').title()}, стр. {s['page']} ({s['file']})" 
                    for s in verified
                ])
                full_response = clean_ans + links
                
                st.markdown(full_response)
                copy_to_clipboard(full_response, "f_copy")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Ошибка API: {e}")
