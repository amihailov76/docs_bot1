import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Verified Pro (Public)", layout="wide")

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
            # Чанки чуть больше, чтобы захватить заголовки разделов
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
        filename = os.path.basename(c.metadata['source'])
        page_num = c.metadata['page'] + 1
        # Метка для поиска в ответе
        label = f"{filename}_p{page_num}"
        
        raw_data.append({"label": label, "content": c.page_content, "file": filename, "page": page_num})
        context_text += f"\n--- ИСТОЧНИК: {label} ---\n{c.page_content}\n"
        
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль (Общий доступ)")
st.info("Режим: Чистый текст + Детальные ссылки. Кнопка 'Показать всё' удалена.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "verified_sources" in m and m["verified_sources"]:
            with st.expander(f"✅ Подтверждающие выдержки (к ответу №{i//2 + 1})"):
                for src in m["verified_sources"]:
                    st.success(f"**Источник: {src['file']}, стр. {src['page']}**")
                    st.text(src['content'])

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
            3. Шаблон ссылки: <Название документа>, <Номер и название раздела>, стр. <номер>, <имя PDF-файла>.
               Пример: Руководство администратора, 5.1 Создание правила, стр. 34, manual.pdf
            4. Если название раздела не найдено в тексте, пиши 'Раздел не указан'.
            5. Название документа бери из самого текста или имени файла.
            
            Для связи источников с выдержками, обязательно в конце каждой строки ссылки в скобках укажи техническую метку файла (например: (FILENAME_pPAGE)).
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # Фильтрация только подтвержденных выдержек
                verified_sources = [s for s in raw_candidates if s['label'] in answer]
                
                # Очищаем ответ от технических меток (типа FILENAME_pPAGE), если ИИ их вывел в скобках
                import re
                clean_answer = re.sub(r'\(.*\.pdf_p\d+\)', '', answer)
                
            except:
                clean_answer = "❌ Ошибка связи с API."
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
