import streamlit as st
import requests

st.title("🔍 Проверка прав API-ключа")
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Ключ не найден в Secrets!")
else:
    # Проверяем список доступных моделей
    url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            st.success("✅ Соединение установлено! Список доступных моделей:")
            for m in models:
                st.write(f"- **ID:** `{m['name']}`")
                st.write(f"  *Описание:* {m['description']}")
                st.write(f"  *Методы:* {m['supportedGenerationMethods']}")
                st.divider()
        else:
            st.error(f"Ошибка {response.status_code}")
            st.json(response.json())
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")
