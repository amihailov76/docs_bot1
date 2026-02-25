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

# --- 2. КОНФИГУРАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if "flash" in m.name: return m.name
        return "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

WORKING_MODEL_NAME = get_working_model()
st.sidebar.write(f"🤖 Движок: `{WORKING_MODEL_NAME}`")

model = genai.GenerativeModel(
    model_name=WORKING_MODEL_NAME,
    generation_config={"temperature": 0.2} # Немного выше для гибкости ответов
)

# --- 3. ОБРАБОТКА PDF ---
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
            # Оптимальный размер чанка для сохранения смысла абзацев
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
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
    
    for c in chunks:
        content_low = c.page_content.lower()
        score = 0
        
        # 1. Огромный вес за точное попадание фразы целиком
        if query_low in content_low:
            score += 100
            
        # 2. Вес за совпадение ключевых слов (длиннее 3 символов)
        for w in query_words:
            if w in content_low:
                score += 10
        
        if score > 0:
            scored.append((score, c))
    
    # Сортировка по релевантности
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Берем топ-7 фрагментов для расширения кругозора модели
    top_chunks = scored[:7]
    
    raw_data = []
    context_text = ""
    for i, (_, c) in enumerate(top_chunks):
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"ID_{i+1}"
        
        # Попытка вытащить заголовок раздела (первая непустая строка)
        lines = [l.strip() for l in c.page_content.split('\n') if len(l.strip()) > 3]
        section = lines[0] if lines else "Техническое описание"
        
        raw_data.append({
            "label": label, 
            "content": c.page_content, 
            "file": filename, 
            "page": page_num,
            "section": section
        })
        context_text += f"\n--- ИСТОЧНИК {label} ---\n{c.page_content}\n"
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Инженерная поддержка MaxPatrol 10")

if st.sidebar.button("Очистить историю"):
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

if prompt := st.chat_input("Введите ваш технический вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализирую техническую документацию..."):
            # Смягченный промпт: поощряем поиск ответа, запрещаем галлюцинации
            system_instruction = (
                "Ты — опытный технический консультант по системе MaxPatrol 10. "
                "Твоя цель: внимательно изучить предоставленные фрагменты текста и найти в них ответ на вопрос пользователя. "
                "Если в тексте есть хотя бы частичная информация — используй её для пошагового объяснения. "
                "Стиль ответа: профессиональный, четкий, без лишних вступлений. "
                "В конце ответа обязательно напиши номера ID всех источников, которые были полезны, в формате: "
                "ИСПОЛЬЗОВАННЫЕ_МЕТКИ: ID_1, ID_2"
            )
            
            try:
                response = model.generate_content(f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}")
                
                if response.text:
                    raw_answer = response.text
                    
                    # Извлекаем метки
                    used_labels = re.findall(r'ID_\d+', raw_answer)
                    verified_sources = [s for s in raw_candidates if s['label'] in used_labels]
                    
                    # Если ИИ не вернул метки, но ответ есть — показываем топ-2 пруфа для уверенности
                    if not verified_sources and raw_candidates:
                        verified_sources = raw_candidates[:2]

                    # Очистка текста
                    clean_answer = re.sub(r'ИСПОЛЬЗОВАННЫЕ_МЕТКИ:.*', '', raw_answer).strip()
                    
                    # Программная сборка ссылок
                    links_block = "\n\n### Ссылки на документацию\n"
                    if verified_sources:
                        for src in verified_sources:
                            doc_title = src['file'].replace('_', ' ').replace('.pdf', '').title()
                            links_block += f"- {doc_title}, {src['section']}, стр. {src['page']}, {src['file']}\n"
                    else:
                        links_block = "\n\n*Ссылки на конкретные разделы не найдены.*"
                    
                    final_answer = clean_answer + links_block
                else:
                    final_answer = "К сожалению, не удалось найти точную инструкцию по вашему запросу."
                    verified_sources = []
            except Exception as e:
                final_answer = f"❌ Ошибка при генерации: {str(e)}"
                verified_sources = []

        st.markdown(final_answer)
        copy_to_clipboard(final_answer, "new_msg")
        
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки", expanded=False):
                for src in verified_sources:
                    st.info(f"**{src['file']}, стр. {src['page']}**\n\n{src['content']}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer, 
            "verified_sources": verified_sources
        })
