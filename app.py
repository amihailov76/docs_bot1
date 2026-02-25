import streamlit as st
import os
import requests
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Verified Pro (Groq)", layout="wide")

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

# --- 2. КОНФИГУРАЦИЯ GROQ API ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
MODEL_ID = "llama-3.3-70b-versatile"  # Самая мощная модель в Groq
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- 3. ВЫБОР ВЕРСИИ (ДОБАВЛЕНО) ---
# Список доступных версий (папок в ./docs)
available_versions = ["27.6", "8.7"]

with st.sidebar:
    st.header("Настройки")
    selected_ver = st.selectbox(
        "Версия MaxPatrol SIEM:",
        available_versions,
        index=0,
        help="Выберите версию системы для поиска в соответствующей документации"
    )

    # Инициализация и сброс истории при смене версии
    if "current_version" not in st.session_state:
        st.session_state.current_version = selected_ver

    if st.session_state.current_version != selected_ver:
        st.session_state.messages = []
        st.session_state.current_version = selected_ver
        st.rerun()

    if st.button("Очистить историю чата"):
        st.session_state.messages = []
        st.rerun()

# --- 4. ОБРАБОТКА PDF (ОБНОВЛЕНО ДЛЯ ВЕРСИЙ) ---
@st.cache_resource
def load_docs_engine(version):
    # Теперь путь строится динамически на основе выбранной версии
    docs_path = os.path.join("./docs", version)
    
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    all_chunks = []
    # Реализация ваших инструкций: сначала adminguide, operatorguide, implementguide
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    all_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    # Сортируем файлы так, чтобы приоритетные были в начале списка обработки
    sorted_files = sorted(
        all_files, 
        key=lambda x: not any(pk in x.lower() for pk in priority_keywords)
    )

    for f in sorted_files:
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в {f}: {e}")
    return all_chunks

# Загружаем чанки только для выбранной версии
chunks = load_docs_engine(selected_ver)

def get_context(query, chunks):
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    
    # Приоритеты из ваших инструкций
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        filename = os.path.basename(c.metadata.get('source', '')).lower()
        
        # Базовый скоринг по словам
        score = sum(2 for w in query_words if len(w) > 3 and w in content_low)
        
        # Повышаем приоритет согласно вашим правилам
        if any(pk in filename for pk in priority_keywords):
            score *= 2.5
            
        if score > 0:
            scored.append((score, c))
            
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:6] # Llama любит качественный, а не избыточный контекст
    
    raw_data = []
    context_text = ""
    for _, c in top_chunks:
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"SOURCE_{filename}_PAGE_{page_num}".replace(" ", "_").replace(".", "_")
        raw_data.append({"label": label, "content": c.page_content, "file": filename, "page": page_num})
        context_text += f"\n--- ИСТОЧНИК_МЕТКА: {label} ---\n{c.page_content}\n"
    return raw_data, context_text

# --- 5. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol SIEM: Помощник пользователя")
st.info(f"""Бот отвечает на вопросы по MaxPatrol SIEM **{selected_ver}**, используя только официальную документацию. С пруфами.  
\n⚠️ **Внимание!** Ответы генерируются ИИ Llama и могут содержать неточности, ошибки, неверные интерпретации документов. Всегда проверяйте важную информацию самостоятельно.  
\nЧтобы не перегрузить бота запросами:  
⏳ Не задавайте более 3-х вопросов в минуту.
&nbsp;  
🎯 Старайтесь формулировать запрос конкретно (например, не "агенты", а "как установить агент в Linux").
&nbsp;  
🧹 После завершения работы очищайте историю чата.&nbsp;""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            copy_to_clipboard(m["content"], f"msg_{i}")
            verified = m.get("verified_sources", [])
            if verified:
                with st.expander(f"✅ Подтверждающие выдержки"):
                    for src in verified:
                        st.success(f"**Источник: {src.get('file')}, стр. {src.get('page')}**")
                        st.text(src.get('content'))

if prompt := st.chat_input(f"Задать вопрос по MaxPatrol SIEM {selected_ver}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Llama 3 анализирует документацию..."):
            system_instruction = """
            Ты — промышленный ИИ-эксперт по системе MaxPatrol 10. Отвечай ТОЛЬКО по предоставленному контексту. 
            Если ответа в тексте нет, вежливо сообщи об этом.
            
            ФОРМАТ ССЫЛОК В КОНЦЕ:
            1. Создай заголовок '### Ссылки на документацию'.
            2. ПИШИ КАЖДУЮ ССЫЛКУ С НОВОЙ СТРОКИ ЧЕРЕЗ ДЕФИС.
            3. Шаблон: - <Название документа>, <Раздел>, стр. <номер>, <имя PDF> (SOURCE_имя_PAGE_номер)
            4. Между текстом ответа и ссылками — двойной отступ.
            
            ОБЯЗАТЕЛЬНО включай метку (SOURCE_...) для каждой ссылки, чтобы я мог верифицировать ответ.
            """
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"КОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}
                ],
                "temperature": 0.1 # Для технической точности
            }
            
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                result = response.json()
                full_response = result['choices'][0]['message']['content']
                
                # Логика верификации источников
                verified_sources = [s for s in raw_candidates if s['label'] in full_response]
                # Убираем технические метки из финального текста для пользователя
                clean_answer = re.sub(r'\(SOURCE_.*?_PAGE_.*?\)', '', full_response)
                
            except Exception as e:
                clean_answer = f"❌ Ошибка Groq API: {str(e)}"
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
