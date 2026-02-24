import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 1. БЕЗОПАСНОСТЬ ---
st.set_page_config(page_title="Corporate Doc Assistant", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль", type="password", on_change=lambda: st.session_state.update({"password_correct": st.session_state.password == "SuperSecret123"}), key="password")
        return False
    return st.session_state["password_correct"]

if not check_password(): st.stop()

# --- 2. ИНИЦИАЛИЗАЦИЯ ---
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

# --- 3. БАЗА ЗНАНИЙ ---
@st.cache_resource
def load_db():
    if not os.path.exists("./docs"): os.makedirs("./docs")
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs: return None
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return Chroma.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

vector_store = load_db()

# --- 4. ЛОГИКА ЧАТА (LCEL - Прямая сборка) ---
if vector_store:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Промпт для генерации ответа
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты технический ассистент. Отвечай на основе контекста:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Функция для форматирования документов
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Собираем цепочку вручную (без использования langchain.chains)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough(), "chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Интерфейс
    if "messages" not in st.session_state: st.session_state.messages = []
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if user_input := st.chat_input("Ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            # Вызываем цепочку
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response)
            
            # Обновляем историю
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.extend([("human", user_input), ("ai", response)])
else:
    st.info("Добавьте PDF в папку /docs и обновите страницу.")
