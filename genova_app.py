import os
import base64
from io import BytesIO
from PIL import Image

import streamlit as st
from groq import Groq
from openai import OpenAI
from huggingface_hub import InferenceClient


# ---------- НАСТРОЙКИ ----------
st.set_page_config(
    page_title="Genova AI — Генератор контента",
    page_icon="🧠",
    layout="wide"
)

# ---------- API КЛЮЧИ ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")

# Инициализация API-клиентов
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None


# ---------- UI: ШАПКА ----------
st.title("🧠 Genova AI — генератор текста и визуала для соцсетей")
st.markdown("Создай **текст, хэштеги и изображение** одним кликом.")

# ---------- UI: ВВОД ДАННЫХ ----------
topic = st.text_input("📝 Тема поста", placeholder="Например: Открытие новой кофейни в центре")

# ⚠️ Обновлённый список платформ
platform = st.selectbox("🌐 Платформа", ["TikTok", "Instagram", "VK", "Telegram", "YouTube"])

tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
sample = st.text_area("📎 Пример поста (по желанию)")

# ---------- ВЫБОР МОДЕЛИ ДЛЯ ТЕКСТА ----------
st.markdown("### 🤖 Модель для текста")
text_model = st.selectbox(
    "Выбери, как генерировать текст",
    ["Groq (бесплатно, LLaMA 3.1)", "OpenAI GPT (платно, GPT-4o)"]
)

# ---------- БЛОК ИЗОБРАЖЕНИЙ ----------
st.markdown("### 🎨 Визуальное оформление")

gen_image = st.checkbox("Сгенерировать изображение")
image_provider = st.selectbox(
    "Провайдер изображения",
    ["Hugging Face (бесплатно, Stable Diffusion 2.1)", "OpenAI DALL·E 3 (платно)"]
)

# Для HF — произвольные размеры; для DALL·E 3 — позже можно добавить поддерживаемые размеры
img_format = st.selectbox("📐 Формат изображения", [
    "512x512",
    "768x512",
    "512x768"
])
image_desc = st.text_input("Текстовый запрос для изображения", placeholder="Например: A neon futuristic city at night with flying cars")


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


def generate_text_openai(prompt: str) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY отсутствует в Secrets.")
    chat = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content


# ---------- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ (HF) ----------
def generate_image_hf(prompt: str, user_size: str) -> Image.Image:
    """
    Генерация картинки через Hugging Face:
    1) генерируем 512x512 (максимально совместимо для бесплатного Inference),
    2) локально масштабируем под выбранный размер.
    """
    if not hf_client:
        raise RuntimeError("HUGGINGFACE_API_KEY отсутствует в Secrets.")
    try:
        raw_bytes = hf_client.text_to_image(
            model="stabilityai/stable-diffusion-2-1",
            prompt=prompt,
            width=512,
            height=512
        )
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        target_w, target_h = map(int, user_size.split("x"))
        if (target_w, target_h) != (512, 512):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        return img
    except Exception as e:
        raise RuntimeError(f"HF error: {e}")


# ---------- КНОПКА: СГЕНЕРИРОВАТЬ ----------
if st.button("🚀 Сгенерировать контент", type="primary"):
    if not topic:
        st.warning("🔔 Пожалуйста, введи тему поста.")
        st.stop()

    # ------ Генерация текста ------
    with st.spinner("⚙️ Генерация текста..."):
        # Небольшая адаптация под платформы
        platform_hint = {
            "TikTok": "Сфокусируйся на коротких, цепляющих фразах и сценарии для видео.",
            "Instagram": "Добавь эмодзи и call-to-action. Можно 3-5 хэштегов в тексте.",
            "VK": "Строй текст информативно, 3–7 хэштегов в конце.",
            "Telegram": "Пиши лаконично, без лишних украшений, можно списком.",
            "YouTube": "Сделай лид-абзац и добавь идеи для описания/тегов."
        }[platform]

        text_prompt = f"""
Ты — помощник по контенту для соцсетей.
Платформа: {platform}. {platform_hint}
Тональность: {tone}
Объем: около {length} слов.
Тема: {topic}
Пример для стилизации: {sample or "нет примера"}.

Верни строго:
1) Текст поста (без приветствий)
2) 5–10 релевантных хэштегов
3) Короткую идею визуала (1–2 предложения)
"""

        try:
            if text_model.startswith("Groq"):
                text_output = generate_text_groq(text_prompt)
            else:
                text_output = generate_text_openai(text_prompt)
        except Exception as e:
            st.error(f"❌ Ошибка генерации текста: {e}")
            st.stop()

    st.markdown("## ✅ Результат")
    st.markdown("### 📝 Текст и хэштеги")
    st.write(text_output)

    # ------ Генерация изображения ------
    if gen_image:
        final_prompt = (image_desc or topic).strip()
        if not final_prompt:
            st.error("❌ Описание изображения пустое.")
        else:
            with st.spinner("🖼 Генерация изображения..."):
                try:
                    if image_provider.startswith("Hugging Face"):
                        img = generate_image_hf(final_prompt, img_format)
                        st.subheader("🖼 Сгенерированное изображение (Hugging Face)")
                        st.image(img, use_column_width=True)
                    else:
                        st.info("DALL·E 3 пока отключён (нужна оплата в OpenAI). Выбери Hugging Face для бесплатной генерации.")
                except Exception as e:
                    st.error(f"🔴 Ошибка генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — текст: Groq/OpenAI, визуал: Hugging Face. Платформы: TikTok, Instagram, VK, Telegram, YouTube.")
