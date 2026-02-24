import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# --- 1. ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant", layout="wide")
target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Вход")
    pwd = st.text_input("Введите пароль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверно")
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY не найден!")
    st.stop()

# Конфигурация с явным указанием транспорта и API
genai.configure(api_key=api_key)

@st.cache_resource
def get_available_model():
    try:
        # Корректно перебираем генератор
        models = list(genai.list_models())
        
        # Сначала ищем Gemini 1.5 Flash
        for m in models:
            if 'gemini-1.5-flash' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        
        # Если нет, ищем любую подходящую модель Gemini
        for m in models:
            if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods:
                return m.name
        
        return "models/gemini-1.5-flash" # Фолбэк
    except Exception as e:
        st.error(f"Ошибка при поиске моделей: {e}")
        return "models/gemini-1.5-flash"

WORKING_MODEL = get_available_model()

# --- 3. ЗАГРУЗКА ТЕКСТА ---
@st.cache_resource
def load_all_text():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        return ""
    
    full_text = ""
    try:
        for f in files:
            loader = PyPDFLoader(os.path.join(docs_path, f))
            pages = loader.load()
            for page in pages:
                full_text += page.page_content + "\n\n"
        return full_text
    except Exception as e:
        return ""

knowledge_base = load_all_text()

# --- 4. ИНТЕРФЕЙС ---
st.title("🤖 Помощник по SIEM")
st.caption(f"Активная модель: {WORKING_MODEL}")

if not knowledge_base:
    st.warning("⚠️ Загрузите PDF в папку docs на GitHub.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Создаем модель
            model = genai.GenerativeModel(WORKING_MODEL)
            
            # Ограничиваем контекст (Gemini 1.5 Flash держит много, но для стабильности возьмем 100к символов)
            context = knowledge_base[:100000] if knowledge_base else "Документация не загружена."
            
            full_prompt = f"Контекст из документов:\n{context}\n\nВопрос пользователя: {prompt}\n\nОтвечай строго по контексту."
            
            response = model.generate_content(full_prompt)
            res_text = response.text
        except Exception as e:
            res_text = f"Ошибка генерации: {str(e)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
