import os
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
from groq import Groq

# ---------- НАСТРОЙКИ ----------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")
st.title("🧠 Genova — AI помощник для соцсетей")

# ---------- КЛЮЧИ API ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))

if not GROQ_API_KEY or not HF_API_KEY:
    st.error("Добавьте API ключи в Streamlit Secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------- UI ----------
topic = st.text_input("📝 Тема поста")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Юмористический"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (необязательно)")

st.markdown("### 🎨 Визуал")
gen_image = st.checkbox("Сгенерировать изображение")
image_prompt = st.text_input("Описание изображения")
format_choice = st.selectbox("Формат:", ["512x512", "768x512", "512x768"])

if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("Введите тему поста.")
        st.stop()

    # ----- ТЕКСТ -----
    with st.spinner("Генерация текста..."):
        try:
            text_prompt = f"""
Ты помощник по созданию контента.
Сгенерируй текст поста по теме "{topic}" для {platform}.
Тональность: {tone}. Длина: {length} слов.
Пример поста (если есть): {sample or 'нет примера'}.
Верни:
1) Сам пост
2) 5–10 хэштегов
3) Предложение для визуала.
"""
            chat = groq_client.chat.completions.create(
                model="llama-3.1-70b-instant",
                messages=[{"role": "user", "content": text_prompt}]
            )
            output = chat.choices[0].message.content
            st.markdown("### ✅ Текст:")
            st.write(output)
        except Exception as e:
            st.error(f"Ошибка Groq API: {e}")

    # ----- ИЗОБРАЖЕНИЕ -----
    if gen_image:
        with st.spinner("Генерация изображения..."):
            try:
                prompt_for_image = image_prompt or topic
                width, height = map(int, format_choice.split("x"))

                headers = {
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                }

                data = {
                    "inputs": prompt_for_image,
                    "parameters": {"width": width, "height": height},
                    "options": {"wait_for_model": True}
                }

                response = requests.post(
                    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                    headers=headers,
                    json=data,
                )

                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.markdown("### 🖼 Сгенерированное изображение")
                    st.image(img, use_column_width=True)
                else:
                    st.error(f"Ошибка HuggingFace API: {response.json()}")

            except Exception as e:
                st.error(f"Ошибка при генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — MVP AI контента для соцсетей (Текст: Groq, Изображения: HuggingFace)")
