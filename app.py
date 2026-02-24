import streamlit as st
import os

# Проверка импортов
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except Exception as e:
    st.error(f"Ошибка загрузки библиотек: {e}")
    st.stop()

# --- 1. ПАРОЛЬ ---
st.set_page_config(page_title="Technical Assistant", layout="wide")

# Берем пароль из секретов или используем дефолтный
target_password = st.secrets.get("COMPANY_PASSWORD", "SuperSecret123")

if "auth" not in st.session_state:
    st.title("🔐 Авторизация")
    pwd = st.text_input("Введите паро_ль", type="password")
    if st.button("Войти"):
        if pwd == target_password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Критическая ошибка: GOOGLE_API_KEY не найден в Secrets!")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def load_docs_and_db():
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    if not files:
        return None

    all_pages = []
    for f in files:
        loader = PyPDFLoader(os.path.join(docs_path, f))
        all_pages.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Используем InMemory базу данных
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3})

retriever = load_docs_and_db()

# --- 3. ИНТЕРФЕЙС ЧАТА ---
st.title("🤖 Технический AI-Ассистент")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Показ истории
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Задайте вопрос по документации..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
        if retriever is None:
            res = "⚠️ Файлы PDF не найдены в папке /docs. Загрузите их в GitHub и перезапустите приложение."
        else:
            prompt_tpl = ChatPromptTemplate.from_template("""
            Ты — корпоративный помощник. Используй только этот текст для ответа:
            {context}
            
            Вопрос: {question}
            """)
            
            chain = (
                {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                 "question": RunnablePassthrough()}
                | prompt_tpl | llm | StrOutputParser()
            )
            
            try:
                res = chain.invoke(prompt)
            except Exception as e:
                res = f"Ошибка нейросети: {e}"
        
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
