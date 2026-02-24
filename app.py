import streamlit as st
import os
import google.generativeai as genai

# 1. ПРОВЕРКА БИБЛИОТЕК
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except Exception as e:
    st.error(f"Ошибка импорта: {e}")
    st.stop()

# --- 2. НАСТРОЙКА И ПАРОЛЬ ---
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

# --- 3. ИНИЦИАЛИЗАЦИЯ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY не найден в Secrets!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Класс-адаптер для эмбеддингов (чтобы Chroma не ругалась на 404)
class GoogleNativeEmbeddings:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return result['embedding']

@st.cache_resource
def load_rag():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None, "Папка /docs пуста."

    try:
        pages = []
        for f in os.listdir(docs_path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_path, f))
                pages.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        # Используем наш кастомный адаптер вместо стандартного LangChain класса
        embeddings = GoogleNativeEmbeddings()
        
        vector_db = Chroma.from_documents(chunks, embeddings)
        return vector_db.as_retriever(search_kwargs={"k": 3}), "База готова!"
    except Exception as e:
        return None, f"Ошибка RAG: {str(e)}"

retriever, status = load_rag()

# --- 4. ЧАТ ---
st.title("🤖 Технический ассистент")
st.caption(status)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ваш вопрос"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            tpl = ChatPromptTemplate.from_template("Используй текст для ответа: {context}\nВопрос: {question}")
            chain = (
                {"context": retriever | (lambda ds: "\n\n".join(d.page_content for d in ds)), 
                 "question": RunnablePassthrough()}
                | tpl | llm | StrOutputParser()
            )
            res = chain.invoke(prompt)
        else:
            res = "Документы не найдены. Проверьте папку /docs в GitHub."
        
        st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
