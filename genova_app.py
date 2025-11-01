import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "YOUR_API_KEY"))

st.title("🧠 Genova — AI помощник для создания контента")
st.markdown("Создавай тексты, ключевые слова и визуальные идеи для любых постов 💡")

topic = st.text_input("📝 Тема поста", placeholder="Например: запуск нового продукта")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (если есть):")

if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
    else:
        with st.spinner("Генерация контента..."):
            prompt = f"""
            Создай пост для соцсетей.
            Платформа: {platform}
            Тональность: {tone}
            Объем: {length} слов
            Пример: {sample or 'нет примера'}
            Тема: {topic}

            Напиши:
            1. Текст поста
            2. Ключевые слова / хэштеги
            3. Идею изображения
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            output = response.choices[0].message.content

        st.markdown("## ✅ Результат:")
        st.write(output)
        st.download_button("💾 Скачать текст", output, file_name="post.txt")

st.markdown("---")
st.caption("🚀 Genova — AI для контент-креаторов (MVP 2025)")

