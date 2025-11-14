import json
import time
import base64
import requests
from io import BytesIO
from PIL import Image

import streamlit as st
from groq import Groq
from openai import OpenAI

# ----------------- НАСТРОЙКИ СТРАНИЦЫ -----------------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")
st.title("🧠 Genova — AI помощник для контента")
st.caption("Тексты + хэштеги + идеи визуала + генерация изображений")

# ----------------- КЛЮЧИ -----------------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

FB_KEY = st.secrets.get("FUSIONBRAIN_API_KEY", "")
FB_SECRET = st.secrets.get("FUSIONBRAIN_API_SECRET", "")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

FUSION_URL = "https://api-key.fusionbrain.ai/"

# ----------------- FUSIONBRAIN API -----------------
def fb_auth_headers():
    if not FB_KEY or not FB_SECRET:
        raise RuntimeError("FusionBrain: не найдены FUSIONBRAIN_API_KEY или FUSIONBRAIN_API_SECRET в Secrets.")
    return {
        "X-Key": f"Key {FB_KEY}",
        "X-Secret": f"Secret {FB_SECRET}",
    }

def fb_get_pipeline_id():
    """Получаем pipeline_id (Kandinsky) как в доке."""
    headers = fb_auth_headers()
    resp = requests.get(FUSION_URL + "key/api/v1/pipelines", headers=headers)
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"FusionBrain /pipelines вернул не JSON:\n{resp.text}")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"FusionBrain /pipelines вернул неожиданный ответ:\n{data}")
    return data[0]["id"]

def fb_generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
    """Генерация изображения через FusionBrain (Kandinsky 3)."""
    headers = fb_auth_headers()
    pipeline_id = fb_get_pipeline_id()

    params = {
        "type": "GENERATE",
        "numImages": 1,
        "width": width,
        "height": height,
        "generateParams": {
            "query": prompt
        }
    }

    files = {
        "pipeline_id": (None, pipeline_id),
        "params": (None, json.dumps(params), "application/json"),
    }

    run_resp = requests.post(FUSION_URL + "key/api/v1/pipeline/run", headers=headers, files=files)

    try:
        run_json = run_resp.json()
    except Exception:
        raise RuntimeError(f"[RUN] FusionBrain вернул не JSON:\n{run_resp.text}")

    if "uuid" not in run_json:
        raise RuntimeError(f"[RUN] Ошибка запуска FusionBrain:\n{run_json}")

    task_id = run_json["uuid"]

    # Ожидаем результат
    status_url = FUSION_URL + f"key/api/v1/pipeline/status/{task_id}"

    for _ in range(60):
        status_resp = requests.get(status_url, headers=headers)
        try:
            status_json = status_resp.json()
        except Exception:
            raise RuntimeError(f"[STATUS] FusionBrain вернул не JSON:\n{status_resp.text}")

        status = status_json.get("status")
        if status == "DONE":
            result = status_json.get("result", {})
            files_list = result.get("files", [])
            if not files_list:
                raise RuntimeError(f"[STATUS] Пустой список files:\n{status_json}")
            img_b64 = files_list[0]
            img_bytes = base64.b64decode(img_b64)
            return Image.open(BytesIO(img_bytes))

        if status == "FAIL":
            err = status_json.get("errorDescription", "Неизвестная ошибка")
            raise RuntimeError(f"FusionBrain: FAIL — {err}")

        time.sleep(1)

    raise RuntimeError("FusionBrain: превышено время ожидания результата.")


# ----------------- ГЕНЕРАЦИЯ ТЕКСТА -----------------
def generate_text_groq(prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY отсутствует в Secrets.")
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

def generate_text_openai(prompt: str) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY отсутствует в Secrets.")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ----------------- UI: ВХОДНЫЕ ДАННЫЕ -----------------
st.subheader("📝 Параметры поста")

topic = st.text_input("Тема поста", placeholder="Например: Открытие новой кофейни в центре")
platform = st.selectbox("🌐 Платформа", ["TikTok", "Instagram", "VK", "Telegram", "YouTube"])
tone = st.selectbox("🎙️ Тональность", ["Дружелюбная", "Официальная", "Мотивирующая", "Юмористическая", "Информационная"])
length = st.slider("📏 Объём текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)")

st.markdown("### 🤖 Модель для текста")
text_model = st.selectbox(
    "Выбери модель",
    ["🆓 Groq — LLaMA 3.1", "💎 OpenAI GPT-4o mini"]
)

st.markdown("### 🎨 Визуальный контент")
gen_image = st.checkbox("Сгенерировать изображение")

image_provider = st.selectbox(
    "Провайдер изображения",
    ["🆓 FusionBrain (Kandinsky)", "💎 OpenAI DALL·E 3 (gpt-image-1)"]
)

image_prompt = st.text_input(
    "Описание изображения",
    placeholder="Например: минималистичная иллюстрация кофейни в тёплых тонах"
)


# ----------------- КНОПКА -----------------
if st.button("🚀 Сгенерировать контент", type="primary"):
    if not topic.strip():
        st.error("Пожалуйста, введи тему поста.")
        st.stop()

    # --- Генерация текста ---
    st.subheader("📄 Текст + хэштеги")

    platform_hint = {
        "TikTok": "Сделай текст в формате идеи для короткого видео + цепкий первый кадр.",
        "Instagram": "Добавь немного эмоций и эмодзи, 3–7 хэштегов в конце.",
        "VK": "Пиши информативно, можно чуть длиннее, 3–7 хэштегов.",
        "Telegram": "Структурировано, можно списками, без лишних эмодзи.",
        "YouTube": "Сделай текст, который подойдёт для описания ролика и закреплённого комментария."
    }[platform]

    text_prompt = f"""
Ты — помощник по созданию контента для соцсетей.

Платформа: {platform}.
Подсказка по оформлению: {platform_hint}
Тональность: {tone}
Желаемый объём: около {length} слов.
Тема: {topic}
Пример для стилизации: {sample or "нет примера"}.

Сгенерируй строго:
1) Текст поста (без приветствия и без лишних пояснений)
2) 5–10 релевантных хэштегов
3) Короткую идею визуала (1–2 предложения)
"""

    with st.spinner("Генерируем текст..."):
        try:
            if text_model.startswith("🆓"):
                text_output = generate_text_groq(text_prompt)
            else:
                text_output = generate_text_openai(text_prompt)
            st.write(text_output)
        except Exception as e:
            st.error(f"❌ Ошибка при генерации текста: {e}")
            st.stop()

    # --- Генерация изображения ---
    if gen_image:
        st.subheader("🖼 Изображение")
        final_prompt = (image_prompt or topic).strip()
        if not final_prompt:
            st.error("Описание изображения пустое. Введи текст или оставь тему поста.")
        else:
            with st.spinner("Генерируем изображение..."):
                # FusionBrain (бесплатно)
                if image_provider.startswith("🆓"):
                    try:
                        img = fb_generate_image(final_prompt, width=1024, height=1024)
                        st.image(img, caption="FusionBrain (Kandinsky)", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ Ошибка FusionBrain: {e}")

                # OpenAI DALL·E 3 / gpt-image-1
                else:
                    if not OPENAI_KEY:
                        st.error("Для DALL·E 3 нужен OPENAI_API_KEY в Secrets.")
                    else:
                        try:
                            resp = openai_client.images.generate(
                                model="gpt-image-1",
                                prompt=final_prompt,
                                size="1024x1024"
                            )
                            img_b64 = resp.data[0].b64_json
                            img_bytes = base64.b64decode(img_b64)
                            img = Image.open(BytesIO(img_bytes))
                            st.image(img, caption="OpenAI gpt-image-1 (DALL·E 3)", use_column_width=True)
                        except Exception as e:
                            st.error(f"❌ Ошибка OpenAI при генерации изображения: {e}")

st.markdown("---")
st.caption("Genova AI — учебный MVP для генерации контента в соцсетях.")
