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

# --- 2. ЖЕСТКАЯ ФИЛЬТРАЦИЯ МОДЕЛИ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Очищаем кэш ресурсов, чтобы принудительно применить новый выбор модели
st.cache_resource.clear()

@st.cache_resource
def force_stable_model():
    """
    Функция принудительного поиска модели 1.5-flash.
    Игнорирует любые упоминания 2.0, 2.5 или experimental.
    """
    try:
        all_models = genai.list_models()
        # Ищем строго 1.5-flash, игнорируя новинки с лимитом 20 запросов
        for m in all_models:
            name = m.name.lower()
            if "1.5-flash" in name and "2.0" not in name and "2.5" not in name and "experimental" not in name:
                # Проверяем поддержку генерации
                if 'generateContent' in m.supported_generation_methods:
                    return m.name
        
        # Если поиск по списку не дал результатов, возвращаем стандартное имя
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

WORKING_MODEL_NAME = force_stable_model()
st.sidebar.warning(f"⚠️ Активный движок: {WORKING_MODEL_NAME}")

model = genai.GenerativeModel(
    model_name=WORKING_MODEL_NAME,
    generation_config={"temperature": 0.1}
)

# --- 3. ОБРАБОТКА PDF С ВЕСАМИ (ВАША ИНСТРУКЦИЯ) ---
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
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {e}")
    return all_chunks

chunks = load_docs_engine()

def get_context(query, chunks):
    if not chunks: return [], ""
    query_low = query.lower()
    query_words = [w for w in query_low.split() if len(w) > 3]
    scored = []
    
    # ПРИОРИТЕТЫ: adminguide, operatorguide, implementguide
    priority_keywords = ['adminguide', 'operatorguide', 'implementguide']
    
    for c in chunks:
        content_low = c.page_content.lower()
        filename_low = os.path.basename(c.metadata.get('source', '')).lower()
        score = 0
        
        # Базовый скоринг текста
        if query_low in content_low:
            score += 100 
        for w in query_words:
            if w in content_low:
                score += 10
        
        # Взвешивание файлов
        is_priority = any(k in filename_low for k in priority_keywords)
        if is_priority:
            score *= 3.0 # Утроенный вес для гайдов
        else:
            score *= 0.6 # Пониженный вес для справочников
        
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:8]
    
    raw_data = []
    context_text = ""
    for i, (_, c) in enumerate(top_chunks):
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"ID_{i+1}"
        
        lines = [l.strip() for l in c.page_content.split('\n') if len(l.strip()) > 5]
        section = lines[0] if lines else "Раздел документации"
        
        raw_data.append({
            "label": label, "content": c.page_content, 
            "file": filename, "page": page_num, "section": section
        })
        context_text += f"\n--- ИСТОЧНИК {label} ---\n{c.page_content}\n"
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ MaxPatrol 10: Verified Engineer")

if st.sidebar.button("Очистить историю и кэш"):
    st.cache_resource.clear()
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and "verified_sources" in m:
            copy_to_clipboard(m["content"], f"h_{i}")
            if m["verified_sources"]:
                with st.expander("✅ Подтверждающие выдержки"):
                    for src in m["verified_sources"]:
                        st.info(f"**{src['file']}, стр. {src['page']}**\n\n{src['content']}")

if prompt := st.chat_input("Ваш технический вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализ документации..."):
            system_instruction = (
                "Ты техподдержка MaxPatrol 10. Отвечай только по контексту. "
                "Если есть шаги настройки — выпиши их. В конце укажи: "
                "ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2"
            )
            
            try:
                response = model.generate_content(f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}")
                
                if response.text:
                    raw_answer = response.text
                    used_labels = re.findall(r'ID_\d+', raw_answer)
                    verified_sources = [s for s in raw_candidates if s['label'] in used_labels]
                    
                    if not verified_sources and raw_candidates:
                        verified_sources = [raw_candidates[0]]

                    clean_answer = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', raw_answer).strip()
                    
                    # Форматирование ссылок
                    links_block = "\n\n### Ссылки на документацию\n"
                    for src in verified_sources:
                        doc_title = src['file'].replace('_', ' ').replace('.pdf', '').title()
                        links_block += f"- {doc_title}, {src['section']}, стр. {src['page']}, {src['file']}\n"
                    
                    final_answer = clean_answer + links_block
                else:
                    final_answer = "Ошибка генерации. Проверьте контекст."
                    verified_sources = []
            except Exception as e:
                final_answer = f"❌ Ошибка API: {str(e)}"
                verified_sources = []

        st.markdown(final_answer)
        copy_to_clipboard(final_answer, "new")
        
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки"):
                for src in verified_sources:
                    st.info(f"**{src['file']}, стр. {src['page']}**\n\n{src['content']}")
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer, "verified_sources": verified_sources})
