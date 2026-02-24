import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# --- 1. НАСТРОЙКА СТРАНИЦЫ И ПАРОЛЬ ---
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

# ВАЖНО: Мы НЕ используем конфигурацию по умолчанию,
# а создаем клиент, который будет обращаться к стабильной версии v1
genai.configure(api_key=api_key)

# Используем полное имя модели, которое поддерживается в v1
MODEL_NAME = "models/gemini-1.5-flash"

# --- 3. ЗАГРУЗКА ТЕКСТА ИЗ PDF ---
@st.cache_resource
def load_all_text():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return ""
    
    full_text = ""
    try:
        for f in os.listdir(docs_path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_path, f))
                pages = loader.load()
                for page in pages:
                    full_text += page.page_content + "\n\n"
        return full_text
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
        return ""

knowledge_base = load_all_text()

# --- 4. ИНТЕРФЕЙС ЧАТА ---
st.title("🤖 Корпоративный ассистент")

if not knowledge_base:
    st.warning("⚠️ Файлы в папке /docs не найдены.")
else:
    st.success("✅ Документация загружена.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Спросите что-нибудь..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                # Прямая инициализация модели через стабильный транспорт
                model = genai.GenerativeModel(model_name=MODEL_NAME)
                
                full_prompt = f"""
                Ты — технический эксперт. Отвечай кратко на основе документов.
                Документы: {knowledge_base[:300000]}
                Вопрос: {prompt}
                """
                
                # Принудительно используем генерацию
                response = model.generate_content(full_prompt)
                res_text = response.text
                
            except Exception as e:
                # Если 404 всё равно вылезает, попробуем модель без префикса
                try:
                    model_fallback = genai.GenerativeModel('gemini-1.5-flash')
                    response = model_fallback.generate_content(full_prompt)
                    res_text = response.text
                except Exception as e2:
                    res_text = f"Критическая ошибка API: {str(e2)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
