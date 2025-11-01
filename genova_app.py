import os
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
from groq import Groq

# ----------------------------------
# НАСТРОЙКИ СТРАНИЦЫ
# ----------------------------------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")
st.title("🧠 Genova — AI помощник для контента")
st.markdown("Создавай тексты, ключевые слова и визуальные идеи. По желанию — генерация изображения (Stable Diffusion).")

# ----------------------------------
# КЛЮЧИ API
# ----------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))

if not GROQ_API_KEY:
    st.error("❗ Не найден GROQ_API_KEY. Добавь его в Secrets.")
if not HF_API_KEY:
    st.error("❗ Не найден HUGGINGFACE_API_KEY. Добавь его в Secrets.")

# Клиент для Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------
# UI: Входные данные
# ----------------------------------
topic = st.text_input("📝 Тема/задача поста", placeholder="Например: Открытие новой кофейни")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)")

st.markdown("### 🎨 Визуал")
gen_image = st.checkbox("Сгенерировать изображение")
image_prompt = st.text_input("Описание изображения (если оставить пустым — возьмём тему поста)")
format_choice = st.selectbox("Формат изображения:", ["Квадрат (512x512)", "Вертикальный (512x768)", "Горизонтальный (768x512)"])

# ----------------------------------
# Кнопка генерации
# ----------------------------------
if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # --- Генерация ТЕКСТА с Groq ---
    with st.spinner("Генерация текста..."):
        try:
            text_prompt = f"""
Ты — помощник по созданию контента для соцсетей.
Сгенерируй текст поста на тему: '{topic}'
Параметры:
- Платформа: {platform}
- Тональность: {tone}
- Длина: {length} слов
- Пример поста (если есть): {sample or 'нет примера'}
Верни:
1) Текст поста
2) 5-10 релевантных ключевых слов/хэштегов
3) Идею визуала
"""
            chat = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": text_prompt}]
            )
            output = chat.choices[0].message.content

            st.markdown("## ✅ Результаты")
            st.markdown("### 📝 Текст и ключевые слова")
            st.write(output)

        except Exception as e:
            st.error(f"Ошибка Groq API: {e}")

    # --- Генерация ИЗОБРАЖЕНИЯ с Hugging Face ---
    if gen_image:
        with st.spinner("Генерация изображения..."):
            try:
                final_img_prompt = (image_prompt or topic).strip()

                # Размеры в зависимости от выбора
                if format_choice == "Квадрат (512x512)":
                    width, height = 512, 512
                elif format_choice == "Вертикальный (512x768)":
                    width, height = 512, 768
                else:
                    width, height = 768, 512

                headers = {
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                    headers=headers,
                    json={
                        "inputs": final_img_prompt,
                        "parameters": {"width": width, "height": height},
                        "options": {"wait_for_model": True},
                    },
                )

                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.markdown("### 🖼 Сгенерированное изображение")
                    st.image(img, use_column_width=True, caption="Stable Diffusion v1.5")
                else:
                    st.error(f"Ошибка HuggingFace API: {response.text}")
            except Exception as e:
                st.error(f"Ошибка при генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — MVP для соцсетей. Текст: Groq Llama. Изображения: Stable Diffusion (Hugging Face).")
