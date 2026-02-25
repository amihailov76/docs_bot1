import streamlit as st
import os
import json
import re
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Engineer Verified Pro", layout="wide")

# CSS для скрытия статуса загрузки, если нужно
st.markdown("""<style> .stDeployButton {display:none;} </style>""", unsafe_allow_html=True)

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

# --- 2. КОНФИГУРАЦИЯ API (ФИКСИРОВАННАЯ) ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Критическая ошибка: GOOGLE_API_KEY не найден в Secrets!")
    st.stop()

genai.configure(api_key=api_key)

# Используем максимально простую и стабильную версию
WORKING_MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name=WORKING_MODEL_NAME)

# --- 3. ОПТИМИЗИРОВАННАЯ ОБРАБОТКА PDF ---
def load_docs_engine():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        st.error(f"Папка {docs_path} не найдена!")
        return []
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        st.warning("В папке /docs нет PDF файлов.")
        return []
    
    all_chunks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, f in enumerate(files):
        status_text.text(f"Обработка: {f}...")
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            # Загружаем страницы по одной, чтобы не забивать память
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {str(e)}")
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("✅ Загрузка завершена!")
    return all_chunks

# --- 4. УПРАВЛЕНИЕ СОСТОЯНИЕМ ---
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    st.header("Управление данными")
    if st.button("🔄 Загрузить/Обновить базу документов"):
        with st.spinner("Индексация документов... это может занять минуту"):
            st.session_state.chunks = load_docs_engine()
            st.success(f"Загружено фрагментов: {len(st.session_state.chunks) if st.session_state.chunks else 0}")
    
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info(f"Движок: `{WORKING_MODEL_NAME}`")

# Логика поиска (вынесена отдельно)
def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    scored = []
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        filename = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for w in query_low.split() if len(w) > 3 and w in content_low)
        
        # Приоритезация по вашему правилу
        if any(k in filename for k in priority_keywords):
            score *= 3.0
        else:
            score *= 0.5
            
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:6]
    
    raw_data = []
    context_text = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        file = os.path.basename(c.metadata.get('source', ''))
        page = c.metadata.get('page', 0) + 1
        raw_data.append({"label": label, "content": c.page_content, "file": file, "page": page})
        context_text += f"\n[{label}]\n{c.page_content}\n"
    return raw_data, context_text

# Отображение чата
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("📚 Источники"):
                for s in m["sources"]:
                    st.caption(f"**{s['file']}, стр. {s['page']}**")

# Ввод запроса
if prompt := st.chat_input("Спросите что-нибудь по MaxPatrol 10..."):
    if not st.session_state.chunks:
        st.error("Сначала загрузите базу документов в боковой панели!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            
            sys_instr = "Ты инженер MP10. Отвечай кратко по контексту. В конце напиши ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1..."
            
            try:
                response = model.generate_content(f"{sys_instr}\n\nКонтекст:\n{context}\n\nВопрос: {prompt}")
                text = response.text
                
                # Чистим метки и собираем ссылки
                clean_text = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', text).strip()
                used_ids = re.findall(r'ID_\d+', text)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                
                links = "\n\n**Ссылки:**\n" + "\n".join([f"- {s['file'].replace('_',' ').title()}, стр. {s['page']}, {s['file']}" for s in verified])
                full_response = clean_text + links
                
                st.markdown(full_response)
                copy_to_clipboard(full_response, "current")
                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": verified})
            except Exception as e:
                st.error(f"Ошибка API: {e}")
