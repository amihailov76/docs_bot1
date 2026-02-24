import streamlit as st
import os

# 1. ПРОВЕРКА БИБЛИОТЕК
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except Exception as e:
    st.error(f"Ошибка импорта: {e}. Проверьте requirements.txt")
    st.stop()

# --- 2. НАСТРОЙКА СТРАНИЦЫ И ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant", layout="wide")

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

# --- 3. ПОДГОТОВКА API И БАЗЫ ЗНАНИЙ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Ошибка: GOOGLE_API_KEY не найден в Secrets!")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def load_docs_and_db():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        return None, "В папке /docs нет PDF-файлов."

    try:
        all_pages = []
        for f in files:
            full_path = os.path.join(docs_path, f)
            loader = PyPDFLoader(full_path)
            all_pages.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(all_pages)
        
        # Использование базовой модели эмбеддингов, которая точно есть в v1beta
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_db = Chroma.from_documents(chunks, embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 3}), "База знаний успешно загружена!"
    except Exception as e:
        return None, f"Ошибка при обработке документов: {str(e)}"

retriever, status_msg = load_docs_and_db()

# --- 4. ИНТЕРФЕЙС ЧАТА ---
st.title("🤖 Технический AI-Ассистент")
st.info(status_msg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Задайте вопрос по документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Используем актуальную модель чата
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        if retriever is None:
            res = "Документы не загружены. Я отвечу на основе общих знаний."
        else:
            prompt_tpl = ChatPromptTemplate.from_template("""
            Используй ТОЛЬКО контекст для ответа. Если ответа нет, скажи, что информации недостаточно.
            Контекст: {context}
            Вопрос: {question}
            """)
            
            chain = (
                {
                    "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                    "question": RunnablePassthrough()
                }
                | prompt_tpl | llm | StrOutputParser()
            )
            
            try:
                with st.spinner("Анализирую..."):
                    res = chain.invoke(prompt)
            except Exception as e:
                res = f"Ошибка API: {str(e)}"
        
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
