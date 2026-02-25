import streamlit as st
import os, json, re, requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. СИСТЕМНЫЕ НАСТРОЙКИ ---
st.set_page_config(page_title="MP10 Verified Engineer", layout="wide")

API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Функция для получения реального списка моделей
@st.cache_resource
def get_available_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            # Фильтруем только те, что поддерживают генерацию контента
            return [m['name'] for m in models_data['models'] if 'generateContent' in m['supportedGenerationMethods']]
        return []
    except:
        return []

# --- 2. ПОДБОР МОДЕЛИ ---
MODELS = get_available_models()
# Ищем 1.5 Flash. Если нет, берем первую попавшуюся Gemini
SELECTED_MODEL = next((m for m in MODELS if "1.5-flash" in m.lower()), None)
if not SELECTED_MODEL and MODELS:
    SELECTED_MODEL = MODELS[0]

# --- 3. ГЕНЕРАЦИЯ (ПРЯМОЙ HTTP) ---
def call_gemini(prompt):
    if not SELECTED_MODEL:
        return "Ошибка: Модели не найдены. Проверьте API ключ."
    
    # ВАЖНО: SELECTED_MODEL уже содержит префикс 'models/'
    url = f"https://generativelanguage.googleapis.com/v1beta/{SELECTED_MODEL}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    try:
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Ошибка API {res.status_code}: {res.text}"
    except Exception as e:
        return f"Ошибка соединения: {str(e)}"

# --- 4. ИНТЕРФЕЙС И ЛОГИКА (БЕЗ ИЗМЕНЕНИЙ) ---
st.title("🏗️ MP10: Verified Engineer")

with st.sidebar:
    st.header("Статус системы")
    if SELECTED_MODEL:
        st.success(f"Подключено к: {SELECTED_MODEL}")
    else:
        st.error("Модели не найдены!")
    
    if st.button("Показать все доступные мне модели"):
        st.write(MODELS)

    if st.button("🔄 Индексировать базу"):
        # ... (код индексации из предыдущих ответов)
        st.info("Индексация запущена...")

# ... (остальной код отображения чата и поиска контекста)
