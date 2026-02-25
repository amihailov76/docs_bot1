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

# --- 2. КОНФИГУРАЦИЯ GOOGLE AI С АВТОПОДБОРОМ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model():
    """Диагностика: ищем модель, которая поддерживает генерацию текста"""
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Отдаем приоритет flash моделям
                if "flash" in m.name:
                    return m.name
        # Если flash не нашли, берем любую первую доступную
        return "gemini-1.5-pro" 
    except Exception as e:
        st.error(f"Не удалось получить список моделей: {e}")
        return "gemini-1.5-flash" # фолбек

WORKING_MODEL_NAME = get_working_model()
st.sidebar.write(f"🤖 Используемая модель: `{WORKING_MODEL_NAME}`")

model = genai.GenerativeModel(
  model_name=WORKING_MODEL_NAME,
  generation_config={"temperature": 0.0}
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
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {e}")
    return all_chunks

chunks = load_docs_engine()

def get_context(query, chunks):
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0: scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:5]
    raw_data = []
    context_text = ""
    for _, c in top_chunks:
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"SOURCE_{filename}_PAGE_{page_num}".replace(" ", "_")
        raw_data.append({"label": label, "content": c.page_content, "file": filename, "page": page_num})
        context_text += f"\n--- ИСТОЧНИК_МЕТКА: {label} ---\n{c.page_content}\n"
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль")

if st.sidebar.button("Очистить историю чата"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            copy_to_clipboard(m["content"], f"hist_{i}")
            verified = m.get("verified_sources", [])
            if verified:
                with st.expander("✅ Подтверждающие выдержки"):
                    for src in verified:
                        st.success(f"**Источник: {src.get('file')}, стр. {src.get('page')}**")
                        st.text(src.get('content'))

if prompt := st.chat_input("Запросить технические данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        with st.spinner("Анализ документации..."):
            system_instruction = "Ты инженерный эксперт. Отвечай только по контексту. В конце сделай список '### Ссылки на документацию' с новой строки через дефис: - Название, Раздел, стр. №, файл (SOURCE_имя_PAGE_номер). Между ответом и ссылками 2 отступа."
            
            try:
                response = model.generate_content(f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}")
                if response.text:
                    full_response = response.text
                    verified_sources = [s for s in raw_candidates if s['label'] in full_response]
                    clean_answer = re.sub(r'\(SOURCE_.*?_PAGE_.*?\)', '', full_response)
                else:
                    clean_answer = "⚠️ Ответ заблокирован или пуст."
                    verified_sources = []
            except Exception as e:
                clean_answer = f"❌ Ошибка ИИ: {str(e)}"
                verified_sources = []

        st.markdown(clean_answer)
        copy_to_clipboard(clean_answer, "new_msg")
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки"):
                for src in verified_sources:
                    st.success(f"**Файл: {src['file']}, Стр: {src['page']}**")
                    st.text(src['content'])
        st.session_state.messages.append({"role": "assistant", "content": clean_answer, "verified_sources": verified_sources})
