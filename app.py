import streamlit as st
import os
import requests
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Verified Pro", layout="wide")

# --- 2. КОНФИГУРАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-flash" 
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_ID}:generateContent?key={api_key}"

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
            # Увеличенный размер чанка, чтобы ИИ видел заголовки разделов
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в {f}: {e}")
            
    return all_chunks

chunks = load_docs_engine()

def get_context(query, chunks):
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:8]
    
    raw_data = []
    context_text = ""
    for _, c in top_chunks:
        filename = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        # Метка для сопоставления (без пробелов для надежности re.sub)
        label = f"SOURCE_{filename}_PAGE_{page_num}".replace(" ", "_")
        
        raw_data.append({
            "label": label, 
            "content": c.page_content, 
            "file": filename, 
            "page": page_num
        })
        context_text += f"\n--- ИСТОЧНИК_МЕТКА: {label} ---\n{c.page_content}\n"
        
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль")

# Кнопка очистки истории (помогает при смене структуры данных)
if st.sidebar.button("Очистить историю чата"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Безопасное отображение источников через .get()
        verified = m.get("verified_sources", [])
        if verified:
            with st.expander(f"✅ Подтверждающие выдержки"):
                for src in verified:
                    f_name = src.get('file', 'Неизвестный файл')
                    p_num = src.get('page', '?')
                    st.success(f"**Источник: {f_name}, стр. {p_num}**")
                    st.text(src.get('content', 'Текст не найден'))

# Поле ввода
if prompt := st.chat_input("Запросить данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        raw_candidates, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализ документации..."):
            system_instruction = """
            Ты — промышленный ИИ-эксперт. Отвечай ТОЛЬКО по контексту. 
            
            ФОРМАТ ССЫЛОК В КОНЦЕ:
            1. Создай раздел '### Ссылки на документацию'.
            2. Каждый источник пиши С НОВОЙ СТРОКИ.
            3. Формат: <Название документа>, <Номер и название раздела>, стр. <номер>, <имя PDF-файла>.
            4. В самом конце строки ссылки ОБЯЗАТЕЛЬНО добавь метку в скобках, например: (SOURCE_filename.pdf_PAGE_12).
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                full_response = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # Фильтруем выдержки: ищем метку label в тексте ответа
                verified_sources = [s for s in raw_candidates if s['label'] in full_response]
                
                # Очищаем финальный текст от технических меток (SOURCE_...)
                clean_answer = re.sub(r'\(SOURCE_.*?_PAGE_.*?\)', '', full_response)
                
            except Exception as e:
                clean_answer = f"❌ Ошибка: {str(e)}"
                verified_sources = []

        st.markdown(clean_answer)
        
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
