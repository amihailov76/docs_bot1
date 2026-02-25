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

# --- 2. КОНФИГУРАЦИЯ GOOGLE AI ---
api_key = st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if "flash" in m.name: return m.name
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

WORKING_MODEL_NAME = get_working_model()
st.sidebar.write(f"🤖 Модель: `{WORKING_MODEL_NAME}`")

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
            splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=300)
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
    for i, (_, c) in enumerate(top_chunks):
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        label = f"ID_{i+1}"
        
        # Предварительно формируем строку ссылки, чтобы ИИ её просто скопировал
        doc_title = filename.replace('.pdf', '').replace('_', ' ').title()
        ref_link = f"{doc_title}, [УКАЖИ_РАЗДЕЛ], стр. {page_num}, {filename}"
        
        raw_data.append({"label": label, "content": c.page_content, "file": filename, "page": page_num})
        context_text += f"\n--- ИСТОЧНИК {label} ---\nССЫЛКА ДЛЯ КОПИРОВАНИЯ: {ref_link}\nТЕКСТ:\n{c.page_content}\n"
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
        if m["role"] == "assistant" and "verified_sources" in m:
            copy_to_clipboard(m["content"], f"hist_{i}")
            if m["verified_sources"]:
                with st.expander("✅ Подтверждающие выдержки"):
                    for src in m["verified_sources"]:
                        st.success(f"**{src['file']}, стр. {src['page']}**")
                        st.text(src['content'])

if prompt := st.chat_input("Запросить технические данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализ документации..."):
            system_instruction = (
                "Ты инженерный эксперт. Отвечай только на основе контекста.\n"
                "ТРЕБОВАНИЯ К ФОРМАТУ ССЫЛОК:\n"
                "1. В конце ответа создай раздел '### Ссылки на документацию'.\n"
                "2. Для каждого использованного источника скопируй строку 'ССЫЛКА ДЛЯ КОПИРОВАНИЯ' из контекста.\n"
                "3. В скопированной строке замени '[УКАЖИ_РАЗДЕЛ]' на название раздела из текста.\n"
                "4. В самый конец каждой строки ссылки добавь метку ID_X (без скобок).\n"
                "5. ЗАПРЕЩЕНО: менять порядок слов в ссылке, использовать кавычки, ставить точки в конце ссылки."
            )
            
            try:
                response = model.generate_content(f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}")
                
                if response.text:
                    full_text = response.text
                    
                    # Сбор пруфов для зеленого блока
                    verified_sources = [s for s in raw_candidates if s['label'] in full_text]
                    
                    # Если ИИ не указал метки, берем топ-1 для надежности
                    if not verified_sources and raw_candidates:
                        verified_sources = [raw_candidates[0]]

                    # ОЧИСТКА: Удаляем технические метки ID_X и лишние артефакты
                    clean_text = re.sub(r'\sID_\d+', '', full_text)
                    final_answer = clean_text.strip()
                else:
                    final_answer = "⚠️ Ответ пуст."
                    verified_sources = []
            except Exception as e:
                final_answer = f"❌ Ошибка: {str(e)}"
                verified_sources = []

        st.markdown(final_answer)
        copy_to_clipboard(final_answer, "new_msg")
        
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки", expanded=False):
                for src in verified_sources:
                    st.success(f"**Файл: {src['file']}, Стр: {src['page']}**")
                    st.text(src['content'])
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer, 
            "verified_sources": verified_sources
        })
