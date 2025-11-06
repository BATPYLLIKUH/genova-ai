import os
import base64
from io import BytesIO
from PIL import Image

import streamlit as st
from huggingface_hub import InferenceClient

# Текстовые провайдеры (по выбору)
from groq import Groq
from openai import OpenAI

# ------------------------------
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ------------------------------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")
st.title("🧠 Genova — AI помощник для контента")
st.caption("Текст: Groq (бесплатно) или OpenAI (ChatGPT, платно) • Изображения: Hugging Face (бесплатно) или DALL·E 3 (платно)")

# ------------------------------
# СЕКРЕТЫ / КЛЮЧИ
# ------------------------------
GROQ_API_KEY      = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
OPENAI_API_KEY    = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))

# Клиенты создадим лениво (когда понадобятся)
groq_client   = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
hf_client     = InferenceClient(api_key=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None

# ------------------------------
# UI — ВХОДНЫЕ ДАННЫЕ
# ------------------------------
colA, colB = st.columns([2, 1])

with colA:
    topic  = st.text_input("📝 Тема/задача поста", placeholder="Например: Открытие новой кофейни")
    sample = st.text_area("📎 Пример поста (по желанию)", placeholder="Вставь пример текста, под который нужно подстроиться")

with colB:
    platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
    tone     = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
    length   = st.slider("📏 Объем текста (слов):", 50, 400, 120)

# Выбор провайдера текста
st.markdown("### 🧠 Провайдер текста")
text_provider = st.radio(
    "Кем генерировать текст?",
    ["Groq (бесплатно)", "OpenAI (ChatGPT, платно)"],
    horizontal=True
)

if text_provider == "Groq (бесплатно)":
    txt_model = st.selectbox("Модель (Groq)", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"])
else:
    txt_model = st.selectbox("Модель (OpenAI)", ["gpt-4o-mini", "gpt-4o"])

# Визуал
st.markdown("### 🎨 Визуальный контент")
gen_image   = st.checkbox("Сгенерировать изображение?")
image_desc  = st.text_input("Описание изображения (если пусто — возьмём тему поста)")

image_provider = st.radio(
    "Кем генерировать изображение?",
    ["Hugging Face (бесплатно)", "OpenAI DALL·E 3 (платно)"],
    horizontal=True
)

# Выбор формата в зависимости от модели
if image_provider == "OpenAI DALL·E 3 (платно)":
    img_format = st.selectbox("📐 Формат изображения", [
        "1024x1024 (квадрат)",
        "1024x1792 (вертикальное)",
        "1792x1024 (горизонтальное)"
    ])
else:
    img_format = st.selectbox("📐 Формат изображения", ["512x512", "768x512", "512x768"])


# ------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------
def generate_text_groq(model: str, prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY отсутствует в Secrets.")
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=900,
    )
    return resp.choices[0].message.content

def generate_text_openai(model: str, prompt: str) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY отсутствует в Secrets.")
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=900,
    )
    return resp.choices[0].message.content

def generate_image_hf(prompt: str, size: str) -> Image.Image:
    if not hf_client:
        raise RuntimeError("HUGGINGFACE_API_KEY отсутствует в Secrets.")
    w, h = map(int, size.split("x"))
    result = hf_client.text_to_image(
        model="stabilityai/stable-diffusion-2",
        prompt=prompt,
        width=w,
        height=h
    )
    return Image.open(BytesIO(result)).convert("RGB")

def generate_image_openai(prompt: str, size: str) -> Image.Image:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY отсутствует в Secrets.")

    size_map = {
        "1024x1024 (квадрат)": "1024x1024",
        "1024x1792 (вертикальное)": "1024x1792",
        "1792x1024 (горизонтальное)": "1792x1024"
    }
    oa_size = size_map.get(size)
    if not oa_size:
        raise ValueError(f"Недопустимый размер изображения для DALL·E 3: {size}")

    resp = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=oa_size
    )
    data = resp.data[0]
    img_bytes = base64.b64decode(data.b64_json)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

# ------------------------------
# КНОПКА: СГЕНЕРИРОВАТЬ
# ------------------------------
if st.button("🚀 Сгенерировать контент", type="primary"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # ---------- ТЕКСТ ----------
    with st.spinner("Генерируем текст..."):
        text_prompt = f"""
Ты — помощник по созданию контента для соцсетей.
Создай текст поста на тему "{topic}".
Параметры:
- Платформа: {platform}
- Тональность: {tone}
- Длина: {length} слов
- Пример: {sample or 'нет примера'}

Выведи строго:
1) Сам пост
2) 5–10 хэштегов
3) Короткую идею визуала (1–2 предложения)
"""
        try:
            if text_provider == "Groq (бесплатно)":
                output = generate_text_groq(txt_model, text_prompt)
            else:
                output = generate_text_openai(txt_model, text_prompt)
            st.subheader("📝 Сгенерированный текст и хэштеги")
            st.write(output)
        except Exception as e:
            st.error(f"🔴 Ошибка генерации текста: {e}")
            st.stop()

    # ---------- ИЗОБРАЖЕНИЕ ----------
    if gen_image:
        with st.spinner("Генерируем изображение..."):
            final_prompt = (image_desc or topic).strip()
            try:
                if image_provider == "Hugging Face (бесплатно)":
                    img = generate_image_hf(final_prompt, img_format)
                    st.subheader("🖼 Сгенерированное изображение (Hugging Face)")
                    st.image(img, use_column_width=True)
                else:
                    img = generate_image_openai(final_prompt, img_format)
                    st.subheader("🖼 Сгенерированное изображение (DALL·E 3)")
                    st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"🔴 Ошибка генерации изображения: {e}")

st.markdown("---")
st.caption("© 2025 • Genova AI — учебный MVP с выбором провайдера (текст/изображения).")
