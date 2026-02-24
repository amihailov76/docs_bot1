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

genai.configure(api_key=api_key)

# ФУНКЦИЯ АВТОПОДБОРА МОДЕЛИ
@st.cache_resource
def get_available_model():
    try:
        # Просим список всех моделей, доступных вашему ключу
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Отдаем приоритет 1.5 flash, если она есть
                if 'gemini-1.5-flash' in m.name:
                    return m.name
        # Если flash не нашли, берем любую первую доступную
        return genai.list_models()[0].name
    except Exception as e:
        st.error(f"Не удалось получить список моделей: {e}")
        return "models/gemini-pro" # Попытка наугад

WORKING_MODEL = get_available_model()

# --- 3. ЗАГРУЗКА ТЕКСТА ---
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
        return ""

knowledge_base = load_all_text()

# --- 4. ИНТЕРФЕЙС ---
st.title("🤖 Корпоративный ассистент")
st.caption(f"Используемая модель: {WORKING_MODEL}")

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
            # Используем найденную модель
            model = genai.GenerativeModel(WORKING_MODEL)
            
            full_prompt = f"Контекст: {knowledge_base[:300000]}\n\nВопрос: {prompt}"
            
            # Принудительная генерация
            response = model.generate_content(full_prompt)
            res_text = response.text
        except Exception as e:
            res_text = f"Ошибка генерации ({WORKING_MODEL}): {str(e)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
