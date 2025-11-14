import streamlit as st
import requests
import time
import base64
from io import BytesIO
from PIL import Image

# ==============================
# FusionBrain: читаем ключи
# ==============================
FB_KEY = st.secrets.get("FUSIONBRAIN_API_KEY", "")
FB_SECRET = st.secrets.get("FUSIONBRAIN_API_SECRET", "")

HEADERS = {
    "X-Key": f"Key {FB_KEY}",
    "X-Secret": f"Secret {FB_SECRET}"
}

# ==============================
# Функция генерации изображения
# ==============================
def generate_fusionbrain(prompt: str):
    run_url = "https://api-key.fusionbrain.ai/key/api/v1/pipeline/run"

    payload = {
        "type": "GENERATE",
        "numImages": 1,
        "width": 1024,
        "height": 1024,
        "generateParams": {
            "query": prompt,
            "steps": 30
        }
    }

    # ---- Запуск генерации ----
    run_resp = requests.post(run_url, headers=HEADERS, json=payload)

    # Проверяем JSON
    try:
        run_json = run_resp.json()
    except Exception:
        raise RuntimeError(f"[RUN] FusionBrain вернул НЕ JSON:\n\n{run_resp.text}")

    if "uuid" not in run_json:
        raise RuntimeError(f"[RUN ERROR] Ответ FusionBrain:\n{run_json}")

    task_id = run_json["uuid"]

    # ---- Проверяем статус ----
    status_url = f"https://api-key.fusionbrain.ai/key/api/v1/pipeline/status/{task_id}"

    for _ in range(60):
        status_resp = requests.get(status_url, headers=HEADERS)

        try:
            status_json = status_resp.json()
        except:
            raise RuntimeError(f"[STATUS] НЕ JSON:\n\n{status_resp.text}")

        if status_json.get("status") == "DONE":
            images = status_json.get("images", [])
            if not images:
                raise RuntimeError("[STATUS] Пустой список images.")

            # Декодируем изображение
            img_bytes = base64.b64decode(images[0])
            return Image.open(BytesIO(img_bytes))

        time.sleep(1)

    raise RuntimeError("FusionBrain: превышено время ожидания результата.")


# ==============================
# UI
# ==============================
st.title("🖼 FusionBrain генератор изображения")

prompt = st.text_input("Описание изображения", placeholder="космическая станция на фоне туманности")

if st.button("Сгенерировать"):
    if not prompt.strip():
        st.error("Введите текстовое описание.")
        st.stop()

    try:
        st.info("⏳ Генерация… ожидаем ответ FusionBrain...")
        image = generate_fusionbrain(prompt)
        st.image(image, caption="Результат FusionBrain", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка FusionBrain:\n\n{e}")
