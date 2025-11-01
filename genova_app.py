import os
import streamlit as st
from openai import OpenAI
import replicate

# ---------- НАСТРОЙКИ СТРАНИЦЫ ----------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")

# ---------- КЛЮЧИ / КЛИЕНТЫ ----------
# Берём ключи из Secrets (Cloud) или локальных переменных среды
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
REPLICATE_TOKEN = st.secrets.get("REPLICATE_API_TOKEN", os.getenv("REPLICATE_API_TOKEN", ""))

# Инициализация OpenAI-клиента
openai_client = OpenAI(api_key=OPENAI_KEY)

# ---------- UI: ШАПКА ----------
st.title("🧠 Genova — AI помощник для соцсетей")
st.markdown("Создавай тексты, ключевые слова и визуальные идеи. И можешь сгенерировать изображение!")

# ---------- UI: ВВОД ----------
topic = st.text_input("📝 Тема поста", placeholder="Например: Открытие новой кофейни")
platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)")

st.markdown("### 🎨 Визуал")
gen_image = st.checkbox("Сгенерировать изображение (Stable Diffusion XL)")
image_prompt = st.text_input("Описание изображения (если оставить пустым — возьмём тему поста)", value="")

# ---------- КНОПКА ----------
if st.button("🚀 Сгенерировать контент"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # ------ Генерация текста ------
    with st.spinner("Генерация текста..."):
        text_prompt = f"""
Ты — помощник по контенту для соцсетей.
Сгенерируй текст для {platform}-поста на тему: "{topic}" в тональности "{tone}".
Объем: около {length} слов.
Пример поста (если есть): {sample}
Также добавь 5–10 релевантных хэштегов и короткую идею визуала.
"""

        chat = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text_prompt}]
        )
        output = chat.choices[0].message.content

    st.markdown("## ✅ Результаты")
    st.markdown("### 📝 Текст и ключевые слова")
    st.write(output)

    # ------ Генерация изображения ------
    if gen_image:
        if not REPLICATE_TOKEN:
            st.error("❗ REPLICATE_API_TOKEN отсутствует. Добавь ключ в Secrets и перезапусти приложение.")
        else:
            with st.spinner("Генерация изображения..."):
                try:
                    final_img_prompt = image_prompt.strip() or topic
                    image_urls = replicate.run(
                        "stability-ai/stable-diffusion-xl-base-1.0",
                        input={"prompt": final_img_prompt}
                    )
                    if isinstance(image_urls, list) and len(image_urls) > 0:
                        st.markdown("### 🖼 Сгенерированное изображение:")
                        st.image(image_urls[0], use_column_width=True, caption="Stable Diffusion XL")
                        st.link_button("🔗 Открыть изображение", image_urls[0])
                    else:
                        st.warning("⚠️ Не удалось получить URL изображения. Попробуй другой промпт.")
                except Exception as e:
                    st.error(f"Ошибка генерации изображения: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🚀 Genova — MVP для соцсетей. Текст: OpenAI GPT. Изображения: Stable Diffusion через Replicate.")
