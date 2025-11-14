import os
import time
import base64
import requests
import streamlit as st
from groq import Groq
from openai import OpenAI
from io import BytesIO
from PIL import Image

# ----------------- Настройки страницы -----------------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")

st.title("🧠 Genova — AI помощник для генерации контента")
st.caption("Создание текста, ключевых слов и изображений под соцсети")


# ----------------- Ключи -----------------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
FB_KEY = st.secrets.get("FUSIONBRAIN_API_KEY", "")
FB_SECRET = st.secrets.get("FUSIONBRAIN_SECRET", "")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


# ----------------- ФУНКЦИЯ: FusionBrain -----------------
def generate_image_fusionbrain(prompt: str, width=512, height=512):
    if not FB_KEY or not FB_SECRET:
        raise RuntimeError("FusionBrain ключи не найдены.")

    url = "https://api.fusionbrain.ai/key/api/v1/text2image/run"

    headers = {
        "X-Key": f"Key {FB_KEY}",
        "X-Secret": f"Secret {FB_SECRET}"
    }

    data = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": 30
    }

    resp = requests.post(url, json=data, headers=headers).json()

    if "uuid" not in resp:
        raise RuntimeError(f"Ошибка запуска FusionBrain: {resp}")

    task_id = resp["uuid"]
    result_url = f"https://api.fusionbrain.ai/key/api/v1/text2image/result?uuid={task_id}"

    for _ in range(40):
        result = requests.get(result_url).json()
        if result.get("status") == "DONE":
            img_bytes = base64.b64decode(result["images"][0])
            return Image.open(BytesIO(img_bytes))
        time.sleep(0.5)

    raise RuntimeError("FusionBrain: превышено время ожидания.")


# ----------------- UI -----------------
st.subheader("📝 Параметры текста")

topic = st.text_input("Тема поста", placeholder="Например: Открытие нового бара...")
platform = st.selectbox("🌐 Платформа", ["TikTok", "Instagram", "VK", "Telegram", "YouTube"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбная", "Официальная", "Мотивирующая", "Юмористическая", "Информационная"])
length = st.slider("📏 Объём текста (слов)", 50, 400, 120)
sample = st.text_area("📎 Пример поста (необязательно)")

# --- выбор модели текста ---
text_model = st.selectbox(
    "🧠 Модель для текста",
    ["🆓 Groq — LLaMA 3.1 (рекомендуется)", "💎 OpenAI GPT-4o mini"]
)


# ---- блок изображений ----
st.subheader("🎨 Генерация изображения")

gen_image = st.checkbox("Создать изображение")

image_model = st.selectbox(
    "🎨 Провайдер изображения",
    ["🆓 FusionBrain", "💎 OpenAI DALL·E 3"]
)

image_prompt = st.text_input("Описание картинки", placeholder="Если пусто — используем тему поста")


# ----------------- Кнопка -----------------
if st.button("🚀 Сгенерировать"):
    if not topic:
        st.error("Введите тему поста.")
        st.stop()

    # ---------- Генерация текста ----------
    st.subheader("📄 Результат")

    with st.spinner("Генерация текста..."):
        text_prompt = f"""
Ты — AI, который пишет тексты для соцсетей.
Платформа: {platform}
Тональность: {tone}
Объём: около {length} слов
Тема: {topic}
Пример: {sample or "нет"}

Сгенерируй:
1) Сам текст поста
2) Список ключевых слов/хэштегов (5–10)
3) Идею визуала (1–2 предложения)
"""

        # Модель Groq
        if text_model.startswith("🆓"):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": text_prompt}]
                )
                text_output = response.choices[0].message.content
            except Exception as e:
                text_output = f"Ошибка Groq: {e}"

        # Модель OpenAI
        else:
            if not OPENAI_KEY:
                text_output = "❌ Нет OpenAI KEY"
            else:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": text_prompt}]
                    )
                    text_output = response.choices[0].message.content
                except Exception as e:
                    text_output = f"Ошибка OpenAI: {e}"

    st.write(text_output)

    # ---------- Генерация изображения ----------
    if gen_image:
        final_img_prompt = image_prompt or topic

        with st.spinner("Генерация изображения..."):

            # FusionBrain
            if image_model.startswith("🆓"):
                try:
                    img = generate_image_fusionbrain(final_img_prompt)
                    st.image(img, caption="FusionBrain")
                except Exception as e:
                    st.error(f"FusionBrain ошибка: {e}")

            # DALL·E 3
            else:
                if not OPENAI_KEY:
                    st.error("Нет OPENAI_API_KEY")
                else:
                    try:
                        resp = openai_client.images.generate(
                            model="gpt-image-1",
                            prompt=final_img_prompt,
                            size="1024x1024"
                        )
                        img_base64 = resp.data[0].b64_json
                        img_bytes = base64.b64decode(img_base64)
                        st.image(Image.open(BytesIO(img_bytes)), caption="OpenAI DALL·E 3")
                    except Exception as e:
                        st.error(f"OpenAI ошибка: {e}")

# ---- Footer ----
st.markdown("---")
st.caption("Genova — AI MVP для генерирования контента в соцсетях")
