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

# --- 2. КОНФИГУРАЦИЯ API И ВЫБОР СТАБИЛЬНОЙ МОДЕЛИ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def get_verified_model():
    """Выбирает 1.5 Flash, строго игнорируя 2.0/2.5 для обхода лимита в 20 запросов"""
    try:
        # Получаем список всех моделей, поддерживающих генерацию
        all_available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. Ищем 1.5 Flash (лимит 1500 запр/день). Исключаем всё, что содержит "2."
        stable_candidates = [
            name for name in all_available 
            if "1.5-flash" in name.lower() and "2." not in name
        ]
        
        if stable_candidates:
            # Приоритет версии с "-latest", если она доступна
            for s in stable_candidates:
                if "latest" in s: return s
            return stable_candidates[0]
            
        # 2. Если 1.5-flash нет, пробуем 1.5-pro (но НЕ 2.x)
        for name in all_available:
            if "1.5-pro" in name.lower() and "2." not in name:
                return name
                
        return "models/gemini-1.5-flash"
    except Exception:
        return "models/gemini-1.5-flash"

ACTIVE_MODEL_NAME = get_verified_model()
model = genai.GenerativeModel(model_name=ACTIVE_MODEL_NAME)

# --- 3. ОБРАБОТКА PDF С ПРИОРИТЕТАМИ ---
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

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

with st.sidebar:
    st.header("Управление")
    if st.button("🔄 Обновить базу PDF"):
        st.session_state.chunks = load_docs_engine()
    
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.write("**Статус системы:**")
    st.code(f"Движок: {ACTIVE_MODEL_NAME}")
    st.caption("Лимит: 1500 запр/день")

def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    scored = []
    
    # ПРИОРИТЕТЫ: adminguide, operatorguide, implementguide
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        filename = os.path.basename(c.metadata.get('source', '')).lower()
        
        # Базовый скоринг релевантности
        score = sum(10 for w in query_low.split() if len(w) > 3 and w in content_low)
        
        # ПРИМЕНЕНИЕ ВЕСОВ ДОКУМЕНТОВ
        if any(k in filename for k in priority_keywords):
            score *= 3.0  # Утроенный вес для гайдов
        else:
            score *= 0.6  # Пониженный вес для рефов и прочих
            
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:7] # Берем 7 лучших фрагментов
    
    raw_data = []
    context_text = ""
    for i, (_, c) in enumerate(top):
        label = f"ID_{i+1}"
        file = os.path.basename(c.metadata.get('source', ''))
        page = c.metadata.get('page', 0) + 1
        raw_data.append({"label": label, "content": c.page_content, "file": file, "page": page})
        context_text += f"\n[{label}]\n{c.page_content}\n"
    return raw_data, context_text

# Отображение истории
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("📚 Источники"):
                for s in m["sources"]:
                    st.caption(f"**{s['file']}, стр. {s['page']}**")

# Ввод вопроса
if prompt := st.chat_input("Введите технический вопрос по MP10..."):
    if not st.session_state.chunks:
        st.warning("Пожалуйста, сначала нажмите кнопку 'Обновить базу PDF' в меню слева.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources, context = get_context(prompt, st.session_state.chunks)
            
            sys_instr = (
                "Ты ведущий инженер техподдержки MaxPatrol 10. Твоя задача — давать точные ответы "
                "на основе предоставленной документации. Если есть пошаговая инструкция, приведи её. "
                "В самом конце ответа укажи только ID: ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2"
            )
            
            try:
                response = model.generate_content(
                    f"{sys_instr}\n\nКонтекст из документов:\n{context}\n\nВопрос пользователя: {prompt}",
                    generation_config={"temperature": 0.1}
                )
                text = response.text
                
                # Обработка ответа и ссылок
                clean_text = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', text).strip()
                used_ids = re.findall(r'ID_\d+', text)
                verified = [s for s in sources if s['label'] in used_ids] or sources[:1]
                
                links = "\n\n**Ссылки на документацию:**\n" + "\n".join([
                    f"- {s['file'].replace('_',' ').title()}, стр. {s['page']}, {s['file']}" 
                    for s in verified
                ])
                full_response = clean_text + links
                
                st.markdown(full_response)
                copy_to_clipboard(full_response, "msg_latest")
                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": verified})
            except Exception as e:
                st.error(f"Ошибка API: {e}")
