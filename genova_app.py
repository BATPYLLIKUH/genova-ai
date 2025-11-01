import os
import streamlit as st
from groq import Groq
import replicate

# ---------- НАСТРОЙКА СТРАНИЦЫ ----------
st.set_page_config(page_title="Genova AI", page_icon="🧠", layout="wide")

# ---------- КЛЮЧИ ----------
GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
REPLICATE_TOKEN = st.secrets.get("REPLICATE_API_TOKEN", os.getenv("REPLICATE_API_TOKEN", ""))

# Проверки ключей (мягко предупреждаем)
if not GROQ_KEY:
    st.warning("⚠️ Не найден GROQ_API_KEY. Добавь его в Secrets, иначе текстовая генерация не заработает.")
if not REPLICATE_TOKEN:
    st.info("ℹ️ REPLICATE_API_TOKEN не задан — генерация изображений будет скрыта.")

# ---------- КЛИЕНТЫ ----------
groq_client = Groq(api_key=GROQ_KEY)

# ---------- UI ----------
st.title("🧠 Genova — AI помощник для соцсетей (бесплатная версия)")
st.markdown("Текст — **Groq (LLaMA 3)**, Изображения — **Stable Diffusion XL (Replicate)**.")

col1, col2 = st.columns([2,1])
with col1:
    topic = st.text_input("📝 Тема/задача поста", placeholder="Например: Открытие новой кофейни в центре")
    sample = st.text_area("📎 Пример поста (по желанию)")
with col2:
    platform = st.selectbox("🌐 Платформа", ["Instagram", "VK", "Telegram", "LinkedIn", "YouTube"])
    tone = st.selectbox("🎙️ Тональность", ["Дружелюбный", "Официальный", "Мотивирующий", "Юмористический", "Информационный"])
    length = st.slider("📏 Объем текста (слов):", 50, 400, 120)
    llm_model = st.selectbox("🧠 Модель текста (Groq)", ["llama3-70b-8192", "llama3-8b-8192"])

st.markdown("### 🎨 Визуал")
gen_image = st.checkbox("Сгенерировать изображение (Stable Diffusion XL)")
image_prompt = st.text_input("Описание изображения (если пусто — возьмём тему поста)", value="")

# ---------- КНОПКА ----------
if st.button("🚀 Сгенерировать контент", type="primary"):
    if not topic:
        st.warning("Пожалуйста, введи тему поста.")
        st.stop()

    # ------ Генерация текста (Groq) ------
    if not GROQ_KEY:
        st.error("Нет GROQ_API_KEY — добавь в Secrets и перезапусти.")
        st.stop()

    with st.spinner("Генерация текста (Groq, LLaMA 3)..."):
        text_prompt = f"""
Ты — помощник по контенту для соцсетей.
Сгенерируй текст для {platform}-поста на тему: "{topic}" в тональности "{tone}".
Объем: около {length} слов.
Если дан пример — подстрой стиль под него.
Пример: {sample or "нет примера"}.

Выведи строго:
1) Текст поста (без лишних приветствий)
2) Список из 5–10 релевантных хэштегов (через пробел или в столбик)
3) Короткую идею визуала (1–2 предложения)
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
    st.markdown("### 📝 Текст и ключевые слова")
    st.write(output)

    # ------ Генерация изображения (Replicate, опционально) ------
    if gen_image:
        if not REPLICATE_TOKEN:
            st.error("❗ REPLICATE_API_TOKEN отсутствует. Добавь ключ в Secrets, чтобы генерировать изображения.")
        else:
            with st.spinner("Генерация изображения (Stable Diffusion XL)..."):
                try:
                    final_img_prompt = (image_prompt or topic).strip()
                    # Вызов SDXL на Replicate
                    image_urls = replicate.run(
                        "stability-ai/stable-diffusion-xl-base-1.0",
                        input={
                            "prompt": final_img_prompt,
                            # Доп.параметры можно включить при желании:
                            # "negative_prompt": "blurry, low quality",
                            # "width": 768, "height": 768,
                            # "num_inference_steps": 30, "guidance_scale": 7.5,
                        }
                    )
                    if isinstance(image_urls, list) and image_urls:
                        url = image_urls[0]
                        st.markdown("### 🖼 Сгенерированное изображение")
                        st.image(url, use_column_width=True, caption="Stable Diffusion XL (Replicate)")
                        st.link_button("🔗 Открыть изображение", url)
                    else:
                        st.warning("Не удалось получить URL изображения. Попробуй уточнить описание.")
                except Exception as e:
                    st.error(f"Ошибка генерации изображения: {e}")

st.markdown("---")
st.caption("🚀 Genova — текст: Groq (LLaMA 3), изображения: Stable Diffusion XL (Replicate). Бесплатный учебный MVP.")
