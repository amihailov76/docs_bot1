import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. НАСТРОЙКА СТРАНИЦЫ И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant PRO", layout="wide")

target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Вход в систему")
    pwd = st.text_input("Введите пароль доступа", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Ошибка: GOOGLE_API_KEY не найден в Secrets!")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_resource
def get_working_model():
    try:
        models = list(genai.list_models())
        for m in models:
            if 'gemini-1.5-flash' in m.name:
                return m.name
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

WORKING_MODEL = get_working_model()

# --- 3. ОБРАБОТКА ДОКУМЕНТОВ (ЧАНКИ) ---
@st.cache_resource
def process_docs():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        return None, "Файлы не найдены в папке /docs."
    
    try:
        all_docs = []
        for f in files:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            all_docs.extend(loader.load())
        
        # РАЗДЕЛЕНИЕ НА ЧАНКИ
        # chunk_size 1500 — это примерно 2-3 абзаца, что идеально для деталей
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(all_docs)
        return chunks, f"Загружено документов: {len(files)}, создано чанков: {len(chunks)}"
    except Exception as e:
        return None, f"Ошибка обработки PDF: {str(e)}"

chunks, status_msg = process_docs()

# --- 4. ФУНКЦИЯ ПОИСКА КОНТЕКСТА ---
def get_relevant_context(query, chunks, top_k=5):
    if not chunks:
        return ""
    
    # Простой, но эффективный поиск по ключевым словам в чанках
    # Это заменяет капризные эмбеддинги и работает мгновенно
    query_words = query.lower().split()
    scored_chunks = []
    
    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk.page_content.lower())
        if score > 0:
            scored_chunks.append((score, chunk.page_content))
    
    # Сортируем по релевантности и берем лучшие чанки
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    relevant_text = "\n\n".join([c[1] for c in scored_chunks[:top_k]])
    
    # Если поиск по словам ничего не дал, берем первые чанки как фолбэк
    if not relevant_text:
        relevant_text = "\n\n".join([c.page_content for c in chunks[:3]])
        
    return relevant_text

# --- 5. ИНТЕРФЕЙС ЧАТА ---
st.title("🤖 Технический ассистент")
st.caption(f"Статус: {status_msg} | Модель: {WORKING_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Спросите детали из документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу информацию в разделах..."):
            try:
                # 1. Извлекаем только нужные куски текста
                context = get_relevant_context(prompt, chunks)
                
                # 2. Формируем запрос к ИИ
                model = genai.GenerativeModel(WORKING_MODEL)
                full_prompt = f"""
                ТЫ: Технический эксперт, который помогает пользователю разобраться в документации.
                ЗАДАЧА: Дай развернутый и конкретный ответ на вопрос, используя выдержки из документов.
                
                ВНИМАНИЕ: 
                - Не пиши просто "Смотри раздел такой-то". 
                - Опиши пошагово, ЧТО именно там написано.
                - Используй списки и технические параметры.
                
                ВЫДЕРЖКИ ИЗ ДОКУМЕНТАЦИИ:
                {context}
                
                ВОПРОС ПОЛЬЗОВАТЕЛЯ:
                {prompt}
                
                ПОДРОБНЫЙ ТЕХНИЧЕСКИЙ ОТВЕТ:
                """
                
                response = model.generate_content(full_prompt)
                res_text = response.text
                
            except Exception as e:
                res_text = f"Ошибка при ответе: {str(e)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
