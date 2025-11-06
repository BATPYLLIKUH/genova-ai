import os
import base64
from io import BytesIO
from PIL import Image

import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient


# ---------- НАСТРОЙКИ ----------
st.set_page_config(
    page_title="Genova AI — Генератор контента",
    page_icon="🧠",
    layout="wide"
)

# ---------- API КЛЮЧИ ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")

# Инициализация клиентов
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None

# ---------- UI: ШАПКА ----------
st.title("🧠 Genova — AI помощник для соцсетей")
st.markdown("Создавай **тексты, ключевые слова и визуальные идеи** бесплатно. Добавлена генерация изображения через Hugging Face.")

# ---------- UI: ВВОД ДАННЫХ ----------
topic = st.text_input("📝 Тема поста", placeholder="Например: Открытие новой кофейни в центре")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)")

st.markdown("### 🎨 Визуальное оформление")

gen_image = st.checkbox("Сгенерировать изображение")
image_provider = st.selectbox(
    "🖼 Провайдер для изображений",
    ["Hugging Face (бесплатно)", "OpenAI DALL·E 3 (платно)"]
)

img_format = st.selectbox("📐 Формат изображения", [
    "512x512",       # базовый 1:1
    "768x512",       # горизонтальный
    "512x768"        # вертикальный
])
image_desc = st.text_input("Описание изображения", placeholder="Например: Futuristic neon city skyline at night")

# ---------- ГЕНЕРАЦИЯ ТЕКСТА ----------
def generate_text_groq(prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY отсутствует в Secrets.")

    chat_resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return chat_resp.choices[0].message.content


# ---------- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ (HF) ----------
def generate_image_hf(prompt: str, user_size: str) -> Image.Image:
    """
    Генерация картинки через Hugging Face (всегда 512x512) → локальное масштабирование в img_format.
    """
    if not hf_client:
        raise RuntimeError("HUGGINGFACE_API_KEY отсутствует в Secrets.")

    base_w, base_h = 512, 512

    try:
        raw_bytes = hf_client.text_to_image(
            model="runwayml/stable-diffusion-v1-5",
            prompt=prompt,
            width=base_w,
            height=base_h
        )
    except Exception as e:
        raise RuntimeError(f"HF text_to_image error: {e}")

    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"HF returned non-image payload: {e}")

    target_w, target_h = map(int, user_size.split("x"))
    if (target_w, target_h) != (base_w, base_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    return img


# ---------- ПРИ НАЖАТИИ КНОПКИ ----------
if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("🔔 Пожалуйста, введи тему поста.")
        st.stop()

    # ------ Генерация текста ------
    with st.spinner("⚙️ Генерация текста..."):
        text_prompt = f"""
Ты — помощник по контенту для соцсетей.
Задача: сгенерировать текст поста.
Параметры:
- Платформа: {platform}
- Тональность: {tone}
- Объем: около {length} слов
- Пример поста: {sample or "нет примера"}
- Тема: {topic}

Сгенерируй:
1) Текст поста с эмодзи (если уместно)
2) 5–10 релевантных хэштегов
3) Короткую идею визуала (1–2 предложения)
"""
        try:
            text_output = generate_text_groq(text_prompt)
        except Exception as e:
            st.error(f"❌ Ошибка Groq API: {e}")
            st.stop()

    st.markdown("## ✅ Результат")
    st.markdown("### 📝 Текст и ключевые слова")
    st.write(text_output)

    # ------ Генерация изображения (опционально) ------
    if gen_image:
        with st.spinner("🖼 Генерация изображения..."):
            final_prompt = (image_desc or topic or "").strip()

            if not final_prompt:
                st.error("❌ Описание изображения пустое.")
            else:
                try:
                    img = generate_image_hf(final_prompt, img_format)
                    st.subheader("🖼 Сгенерированное изображение (Hugging Face)")
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"🔴 Ошибка генерации изображения: {e}")


# ---------- ФУТЕР ----------
st.markdown("---")
st.caption("🚀 Genova — MVP для соцсетей. Текст: Groq (Llama 3). Изображения: Hugging Face.")
