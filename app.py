import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА И БЕЗОПАСНОСТЬ ---
st.set_page_config(page_title="Engineer Verified Assistant", layout="wide")

# Проверка пароля
target_password = st.secrets.get("COMPANY_PASSWORD", "123")
if "auth" not in st.session_state:
    st.title("🔐 Авторизация системы")
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

# --- 3. ПОДГОТОВКА БАЗЫ ЗНАНИЙ ---
@st.cache_resource
def load_docs_with_metadata():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    all_chunks = []
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    for f in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            # Инженерный сплиттер: куски по 1200 знаков для точности
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            st.error(f"Ошибка в файле {f}: {e}")
            
    return all_chunks

chunks = load_docs_with_metadata()

def get_context_for_query(query, chunks):
    """Находит наиболее релевантные куски текста для конкретного вопроса"""
    if not chunks: return [], ""
    query_words = query.lower().split()
    scored = []
    for c in chunks:
        score = sum(1 for w in query_words if w in c.page_content.lower())
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:6] # Берем топ-6 самых точных совпадений
    
    context_text = ""
    raw_sources_data = []
    
    for _, c in top_chunks:
        source_name = os.path.basename(c.metadata.get('source', 'unknown'))
        page_num = c.metadata.get('page', 0) + 1
        header = f"Файл: {source_name}, Стр: {page_num}"
        
        context_text += f"\n--- ИСТОЧНИК: {header} ---\n{c.page_content}\n"
        raw_sources_data.append({"header": header, "content": c.page_content})
        
    return raw_sources_data, context_text

# --- 4. ИНТЕРФЕЙС ЧАТА ---
st.title("🏗️ Технический контроль документации")
st.caption(f"Активная модель: {MODEL_ID} | Режим верификации: Включен")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории сообщений
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Показываем блок верификации ТОЛЬКО если он привязан к этому конкретному сообщению
        if "raw_sources" in m and m["raw_sources"]:
            with st.expander(f"🔍 Исходные данные из PDF для ответа №{i//2 + 1}"):
                for idx, src in enumerate(m["raw_sources"]):
                    st.info(f"**{src['header']}**")
                    st.text(src['content']) # Используем text для сохранения форматирования PDF

# Ввод нового вопроса
if prompt := st.chat_input("Запросить технические данные..."):
    # Сохраняем и отображаем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Поиск контекста именно для ТЕКУЩЕГО вопроса
        current_raw_sources, context_for_ai = get_context_for_query(prompt, chunks)
        
        with st.spinner("Сверка с базой регламентов..."):
            system_instruction = """
            ТЫ: Промышленный ИИ-эксперт.
            ЗАДАЧА: Дать ответ ТОЛЬКО на основе предоставленного КОНТЕКСТА.
            
            ТРЕБОВАНИЯ К ОФОРМЛЕНИЮ:
            1. Пиши только техническую суть. БЕЗ ссылок внутри предложений.
            2. Если информации нет в контексте, ответь: "В загруженных спецификациях данные отсутствуют".
            3. В КОНЦЕ ответа создай раздел "### Ссылки на документацию" и перечисли там все файлы и страницы.
            4. Температура ответа: 0.0 (строгая точность).
            """
            
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nКОНТЕКСТ:\n{context_for_ai}\n\nВОПРОС:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 3000}
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=40)
                if response.status_code == 200:
                    answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    answer = f"❌ Ошибка API ({response.status_code}). Проверьте ключ или модель."
            except Exception as e:
                answer = f"❌ Ошибка связи с сервером: {str(e)}"

        # Вывод ответа
        st.markdown(answer)
        
        # Вывод блока верификации (Expander) сразу под ответом
        if current_raw_sources:
            with st.expander("🔍 Исходные данные из PDF для этого ответа"):
                for idx, src in enumerate(current_raw_sources):
                    st.info(f"**{src['header']}**")
                    st.text(src['content'])
        
        # Сохранение в историю с жесткой привязкой источников
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "raw_sources": current_raw_sources # Важно: сохраняем список источников внутри объекта сообщения
        })
