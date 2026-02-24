import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. НАСТРОЙКА СТРАНИЦЫ И БЕЗОПАСНОСТЬ ---
st.set_page_config(page_title="Technical AI Assistant", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.title("🔐 Доступ ограничен")
        st.text_input(
            "Введите пароль", 
            type="password", 
            on_change=lambda: st.session_state.update({"password_correct": st.session_state.password == "SuperSecret123"}), 
            key="password"
        )
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ API ---
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

# --- 3. БАЗА ЗНАНИЙ (RAG) ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("./docs"):
        os.makedirs("./docs")
    
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma.from_documents(chunks, embeddings)

vector_store = load_knowledge_base()

# --- 4. ИНТЕРФЕЙС И ЛОГИКА ЧАТА ---
st.title("🤖 Технический ассистент")

if vector_store:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Определение промпта
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Ты эксперт по технической документации. Отвечай кратко и точно, используя предоставленный контекст:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Форматирование документов в строку
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Сборка цепочки LCEL (Pipe-синтаксис)
    # Это заменяет ConversationalRetrievalChain и работает стабильнее
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "input": RunnablePassthrough(), 
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Работа с историей сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Отрисовка сообщений из истории
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Ввод вопроса пользователем
    if user_query := st.chat_input("Задайте вопрос по документации..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                # Вызов цепочки
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(response)
            
            # Сохраняем в историю
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.extend([("human", user_query), ("ai", response)])
            
            # Ограничиваем историю 10 записями
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
else:
    st.warning("📂 Папка 'docs' пуста. Загрузите PDF-файлы в репозиторий GitHub.")
