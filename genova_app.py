import os
import requests
import streamlit as st
from groq import Groq
from PIL import Image
from io import BytesIO

# ---------- НАСТРОЙКА СТРАНИЦЫ ----------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")

# ---------- КЛЮЧИ ----------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", ""))

# Проверка ключей
if not GROQ_KEY:
    st.warning("⚠️ Не найден GROQ_API_KEY. Добавь его в Secrets, иначе текстовая генерация не заработает.")

if not HF_API_KEY:
    st.warning("⚠️ Не найден HUGGINGFACE_API_KEY. Добавь его в Secrets, иначе генерация изображений не заработает.")

# ---------- КЛИЕНТЫ ----------
groq_client = Groq(api_key=GROQ_KEY)

# ---------- UI ----------
st.title("🧠 Genova — AI помощник для соцсетей (бесплатная версия)")
st.markdown("Текст — **Groq (LLaMA / Mixtral)**, Изображения — **Stable Diffusion 2.1 (Hugging Face)**.")

col1, col2 = st.columns([2, 1])
with col1:
    topic = st.text_input("📝 Тема/задача поста", placeholder="Например: Открытие новой кофейни в центре")
    sample = st.text_area("📎 Пример поста (по желанию)", placeholder="Можешь вставить текст, под который надо подстроиться...")
with col2:
    platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
    tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
    length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
    llm_model = st.selectbox("🧠 Модель текста (Groq)", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"])

# Генерация изображения: параметры
st.markdown("### 🎨 Визуал")
gen_image = st.checkbox("Сгенерировать изображение")
image_prompt = st.text_input("Описание изображения (если пусто — возьмём тему поста)", value="")

format_choice = st.selectbox(
    "🖼 Формат изображения",
    ["Квадрат (512x512)", "Вертикальный (512x768)", "Горизонтальный (768x512)"]
)

# ---------- КНОПКА ----------
if st.button("🚀 Сгенерировать контент", type="primary"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # ------ Генерация текста (Groq) ------
    with st.spinner("Генерация текста..."):
        text_prompt = f"""
Ты — помощник по контенту для соцсетей.
Сгенерируй текст для {platform}-поста на тему: "{topic}" в тональности "{tone}".
Объем: около {length} слов.
Пример текста: {sample or "нет примера"}.

Выведи:
1) Текст поста (без приветствий)
2) 5–10 хэштегов
3) Идея визуала (коротко)
"""
        try:
            chat = groq_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": text_prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            output = chat.choices[0].message.content
        except Exception as e:
            st.error(f"Ошибка Groq API: {e}")
            st.stop()

    st.markdown("## ✅ Результаты")
    st.markdown("### 📝 Текст и хэштеги")
    st.write(output)

    # ------ Генерация изображения (Hugging Face) ------
    if gen_image:
        if not HF_API_KEY:
            st.error("❗ HUGGINGFACE_API_KEY отсутствует. Добавь его в Secrets.")
        else:
            with st.spinner("Генерация изображения..."):
                try:
                    final_img_prompt = (image_prompt or topic).strip()

                    # Выбор размеров
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
                        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
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
                        st.image(img, use_column_width=True, caption="По модели Stable Diffusion 2.1")
                    else:
                        st.error(f"Ошибка HuggingFace API: {response.text}")

                except Exception as e:
                    st.error(f"Ошибка при генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — на Groq + Hugging Face (Stable Diffusion). Полностью бесплатный MVP.")
