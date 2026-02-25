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

# --- 2. УМНАЯ ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (ФИКС 404 И 429) ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Критическая ошибка: GOOGLE_API_KEY не найден!")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model_safely():
    """Находит правильное имя модели, поддерживаемое вашим API ключом"""
    try:
        # Получаем список всех моделей
        all_models = genai.list_models()
        # Фильтруем: нужна 1.5 Flash (высокая квота), но НЕ 2.5 (низкая квота)
        candidates = [
            m.name for m in all_models 
            if 'generateContent' in m.supported_generation_methods 
            and "1.5-flash" in m.name 
            and "2.5" not in m.name
        ]
        if candidates:
            # Возвращаем первое найденное имя (обычно это models/gemini-1.5-flash)
            return candidates[0]
        return "gemini-1.5-flash" # Резервный вариант
    except Exception:
        return "gemini-1.5-flash"

ACTIVE_MODEL_NAME = get_working_model_safely()
model = genai.GenerativeModel(model_name=ACTIVE_MODEL_NAME)

# --- 3. ОБРАБОТКА PDF ---
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
        status_text.text(f"Индексация: {f}...")
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {str(e)}")
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("✅ База знаний готова!")
    return all_chunks

# --- 4. УПРАВЛЕНИЕ СОСТОЯНИЕМ ---
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    st.header("База знаний")
    if st.button("🔄 Обновить документы"):
        with st.spinner("Загрузка..."):
            st.session_state.chunks = load_docs_engine()
    
    if st.button("🗑️ Очистить историю"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info(f"Используется: `{ACTIVE_MODEL_NAME}`")

def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    scored = []
    # Ваши приоритеты из сохраненной инструкции
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        filename = os.path.basename(c.metadata.get('source', '')).lower()
        score = sum(10 for w in query_low.split() if len(w) > 3 and w in content_low)
        
        # Приоритезация по именам файлов
        if any(k in filename for k in priority_keywords):
            score *= 3.0
        else:
            score *= 0.6
            
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
if prompt := st.chat_input("Настроить группу активов..."):
    if not st.session_state.chunks:
        st.warning("Пожалуйста, сначала нажмите 'Обновить документы' в меню слева.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            
            sys_instr = (
                "Ты технический инженер MP10. Отвечай только на основе контекста. "
                "Если в контексте есть инструкция, напиши её пошагово. "
                "В конце напиши: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2..."
            )
            
            try:
                response = model.generate_content(
                    f"{sys_instr}\n\nКонтекст:\n{context}\n\nВопрос: {prompt}",
                    generation_config={"temperature": 0.1}
                )
                text = response.text
                
                clean_text = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', text).strip()
                used_ids = re.findall(r'ID_\d+', text)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                
                links = "\n\n**Ссылки:**\n" + "\n".join([
                    f"- {s['file'].replace('_',' ').title()}, стр. {s['page']}, {s['file']}" 
                    for s in verified
                ])
                full_response = clean_text + links
                
                st.markdown(full_response)
                copy_to_clipboard(full_response, "cur")
                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": verified})
            except Exception as e:
                st.error(f"Ошибка API: {e}. Попробуйте обновить страницу.")
