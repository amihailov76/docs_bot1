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
    st.error("GOOGLE_API_KEY не найден в Secrets!")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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
    st.warning("⚠️ Файлы в папке /docs не найдены или они пусты.")
else:
    st.success("✅ Документация загружена и готова к анализу.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Показываем историю
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
                # Мы просто отправляем ВЕСЬ текст документов как контекст. 
                # Gemini 1.5 Flash легко переваривает до 1 млн токенов.
                full_prompt = f"""
                Ты — технический эксперт. Используй следующую документацию для ответа на вопрос.
                Если в документации нет ответа, ответь на основе своих знаний, но предупреди об этом.
                
                ДОКУМЕНТАЦИЯ:
                {knowledge_base[:500000]} 
                
                ВОПРОС:
                {prompt}
                """
                
                response = model.generate_content(full_prompt)
                res_text = response.text
            except Exception as e:
                res_text = f"Произошла ошибка: {str(e)}"
        
        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
