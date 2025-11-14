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

st.title("🧠 Genova — AI генератор контента")
st.caption("Тексты + изображения (FusionBrain / DALL·E 3)")


# ----------------- Ключи -----------------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

FB_KEY = st.secrets.get("FUSIONBRAIN_API_KEY", "")
FB_SECRET = st.secrets.get("FUSIONBRAIN_SECRET", "")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


# ----------------- FusionBrain DEBUG-ФУНКЦИЯ -----------------
def fusionbrain_generate_debug(prompt: str):

    url = "https://api.fusionbrain.ai/key/api/v1/text2image/run"

    headers = {
        "X-Key": f"Key {FB_KEY}",
        "X-Secret": f"Secret {FB_SECRET}"
    }

    data = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "num_steps": 30
    }

    # -----------------------
    # 1) Первый запрос — запуск задачи
    # -----------------------
    raw = requests.post(url, json=data, headers=headers)

    st.write("### 🟦 FusionBrain RAW response (run)")
    st.code(raw.text)

    # Если ответ не JSON — пишем ошибку
    try:
        resp = raw.json()
    except:
        raise RuntimeError("FusionBrain вернул НЕ JSON. Это обычно означает: неверные ключи или формат заголовков.")

    if "uuid" not in resp:
        raise RuntimeError(f"Ошибка запуска FusionBrain: {resp}")

    task_id = resp["uuid"]

    # -----------------------
    # 2) Ждём результат
    # -----------------------
    result_url = f"https://api.fusionbrain.ai/key/api/v1/text2image/result?uuid={task_id}"

    st.write("### ⏳ FusionBrain ожидаем результат...")

    for _ in range(60):
        result_raw = requests.get(result_url)

        st.write("### 🟩 FusionBrain RAW response (result)")
        st.code(result_raw.text)

        try:
            result = result_raw.json()
        except:
            raise RuntimeError("FusionBrain result вернул не JSON.")

        if result.get("status") == "DONE":
            img_bytes = base64.b64decode(result["images"][0])
            return Image.open(BytesIO(img_bytes))

        time.sleep(0.5)

    raise RuntimeError("FusionBrain: задача слишком долго обрабатывается")


# ----------------- UI -----------------
st.subheader("📝 Параметры текста")

topic = st.text_input("Тема поста")
platform = st.selectbox("Платформа", ["TikTok", "Instagram", "VK", "Telegram", "YouTube"])
tone = st.selectbox("Тональность", ["Дружелюбная", "Официальная", "Мотивирующая", "Юмористическая", "Информационная"])
length = st.slider("Объём (слов):", 50, 400, 120)
sample = st.text_area("Пример поста (опционально)")

text_model = st.selectbox(
    "🧠 Модель текста",
    ["🆓 Groq — LLaMA 3.1", "💎 OpenAI GPT-4o mini"]
)

st.subheader("🎨 Генерация изображения")
gen_image = st.checkbox("Создать изображение")

image_model = st.selectbox(
    "🎨 Провайдер",
    ["🆓 FusionBrain", "💎 OpenAI DALL·E 3"]
)

image_prompt = st.text_input("Описание изображения")


# ----------------- Генерация -----------------
if st.button("🚀 Сгенерировать"):
    if not topic:
        st.error("Введите тему!")
        st.stop()

    # ---------- ТЕКСТ ----------
    st.subheader("📄 Текст")

    text_prompt = f"""
Платформа: {platform}
Тональность: {tone}
Длина: {length} слов
Тема: {topic}
Пример: {sample or "нет"}

Сгенерируй:
1) Текст поста
2) 5-10 хэштегов
3) Идею визуала
"""

    with st.spinner("Генерация текста..."):

        if text_model.startswith("🆓"):
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": text_prompt}]
            )
            text_output = response.choices[0].message.content

        else:
            if not OPENAI_KEY:
                text_output = "❌ NET OPENAI KEY"
            else:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": text_prompt}]
                )
                text_output = response.choices[0].message.content

    st.write(text_output)

    # ---------- ИЗОБРАЖЕНИЕ ----------
    if gen_image:
        st.subheader("🖼 Изображение")

        final_prompt = image_prompt or topic

        if image_model.startswith("🆓"):
            try:
                img = fusionbrain_generate_debug(final_prompt)
                st.image(img, caption="FusionBrain")
            except Exception as e:
                st.error(f"FusionBrain ошибка: {e}")

        else:
            if not OPENAI_KEY:
                st.error("Нет OpenAI ключа")
            else:
                try:
                    res = openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=final_prompt,
                        size="1024x1024"
                    )
                    img_bytes = base64.b64decode(res.data[0].b64_json)
                    st.image(Image.open(BytesIO(img_bytes)), caption="DALL·E 3")
                except Exception as e:
                    st.error(f"OpenAI ошибка: {e}")

st.markdown("---")
st.caption("Genova AI — MVP для генерации контента")
