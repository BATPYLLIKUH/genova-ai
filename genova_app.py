import os
import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

# ------------------------------
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ------------------------------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")
st.title("🧠 Genova — AI помощник для создания контента в соцсетях")

# ------------------------------
# ЗАГРУЗКА СЕКРЕТОВ
# ------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))

# Проверка ключей
if not GROQ_API_KEY:
    st.error("❗ GROQ_API_KEY отсутствует. Добавь его в Secrets.")
    st.stop()
if not HF_API_KEY:
    st.error("❗ HUGGINGFACE_API_KEY отсутствует. Добавь его в Secrets.")
    st.stop()

# Инициализация клиентов
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(api_key=HF_API_KEY)

# ------------------------------
# UI — ВВОДНЫЕ ДАННЫЕ
# ------------------------------
topic = st.text_input("📝 Тема/задача поста", placeholder="Например: Открытие новой кофейни")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
tone = st.selectbox("🎙️ Тональность текста", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)", placeholder="Необязательный пример для ориентации модели")

# Выбор модели текста
model_choice = st.selectbox("🧠 Модель текста (Groq)", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])

# Генерация изображения
st.markdown("### 🎨 Визуальный контент")
gen_image = st.checkbox("Хочу сгенерировать изображение")
image_prompt = st.text_input("Описание изображения (если не заполнить — возьмем тему поста)")
format_choice = st.selectbox("📐 Формат изображения:", ["512x512", "768x512", "512x768"])

# ------------------------------
# КНОПКА СТАРТА
# ------------------------------
if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # ---------- Генерация текста ----------
    with st.spinner("Генерация текста с Groq..."):
        try:
            prompt = f"""
Ты — помощник по созданию контента для соцсетей.
Создай текст поста на тему "{topic}".
Параметры:
- Платформа: {platform}
- Тональность: {tone}
- Длина: {length} слов
- Пример: {sample or 'нет примера'}
Верни:
1) Сам пост
2) 5–10 хэштегов
3) Идею визуала
"""
            chat = groq_client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}]
            )
            output = chat.choices[0].message.content
            st.markdown("### ✅ Сгенерированный текст и хэштеги:")
            st.write(output)
        except Exception as e:
            st.error(f"Ошибка Groq API: {e}")

    # ---------- Генерация изображения ----------
    if gen_image:
        with st.spinner("Генерация изображения с Hugging Face..."):
            try:
                img_prompt = image_prompt.strip() or topic
                width, height = map(int, format_choice.split("x"))

                result = hf_client.text_to_image(
                    prompt=img_prompt,
                    model="runwayml/stable-diffusion-v1-5",
                    width=width,
                    height=height
                )

                image = Image.open(BytesIO(result)).convert("RGB")
                st.markdown("### 🖼 Сгенерированное изображение:")
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error(f"Ошибка при генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — AI MVP для генерации контента. Текст: Groq. Изображения: Hugging Face.")
