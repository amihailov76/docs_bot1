import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА ---
st.set_page_config(page_title="Engineer Verified Pro", layout="wide")

# Проверка пароля
target_password = st.secrets.get("COMPANY_PASSWORD", "123")
if "auth" not in st.session_state:
    st.title("🔐 Авторизация")
    pwd = st.text_input("Введите инженерный пароль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Доступ запрещен")
    st.stop()

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
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
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
    top_chunks = scored[:8] # Берем до 8 кандидатов
    
    raw_data = []
    context_text = ""
    for _, c in top_chunks:
        header = f"Файл: {os.path.basename(c.metadata['source'])}, Стр: {c.metadata['page'] + 1}"
        raw_data.append({"header": header, "content": c.page_content})
        context_text += f"\n--- {header} ---\n{c.page_content}\n"
        
    return raw_data, context_text

# --- 4. ИНТЕРФЕЙС ---
st.title("🏗️ Технический контроль (Strict & Full View)")
st.info("Бот фильтрует исходники согласно финальным ссылкам. Полный лог доступен по запросу.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        if "verified_sources" in m and m["verified_sources"]:
            with st.expander(f"✅ Подтверждающие выдержки (к ответу №{i//2 + 1})"):
                for src in m["verified_sources"]:
                    st.success(f"**{src['header']}**")
                    st.text(src['content'])
            
            with st.expander(f"⚙️ Показать всё найденное (RAW Context)"):
                for src in m["all_found"]:
                    st.caption(f"**{src['header']}**")
                    st.text(src['content'])

# Ввод
if prompt := st.chat_input("Запросить данные..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Поиск всех кандидатов
        all_found, context_for_ai = get_context(prompt, chunks)
        
        with st.spinner("Анализ и фильтрация источников..."):
            system_instruction = """
            Ты — промышленный ИИ-эксперт. 
            Отвечай ТОЛЬКО по контексту. 
            В конце ВСЕГДА пиши раздел '### Ссылки на документацию'.
            Каждая ссылка должна быть строго в формате: 'Файл: [имя], Стр: [номер]'.
            Не используй ссылки в тексте, только в финальном списке.
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # 2. ФИЛЬТРАЦИЯ: оставляем только те чанки, которые ИИ упомянул в тексте
                verified_sources = [s for s in all_found if s['header'] in answer]
            except:
                answer = "❌ Ошибка связи с API."
                verified_sources = []

        st.markdown(answer)
        
        # 3. Вывод блоков сразу
        if verified_sources:
            with st.expander("✅ Подтверждающие выдержки"):
                for src in verified_sources:
                    st.success(f"**{src['header']}**")
                    st.text(src['content'])
        
        with st.expander("⚙️ Показать всё найденное"):
            for src in all_found:
                st.caption(f"**{src['header']}**")
                st.text(src['content'])
        
        # Сохранение
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "verified_sources": verified_sources,
            "all_found": all_found
        })
