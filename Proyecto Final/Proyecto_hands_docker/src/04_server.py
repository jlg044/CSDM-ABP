"""
04_server.py
------------
Servidor de inferencia en tiempo real (corre en Docker).

Flujo:
  - Carga best_model.keras una sola vez al arrancar
  - Expone POST /predict: recibe un PNG del esqueleto normalizado
    y devuelve JSON con la clase predicha, número de dedos y confianza
  - Expone GET /health para que el cliente verifique disponibilidad

Arrancar desde docker-compose (servicio inference-server):
    docker compose up inference-server

Arrancar manualmente dentro del contenedor:
    uvicorn src.04_server:app --host 0.0.0.0 --port 8000
"""

import io
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ── Constantes (deben coincidir con 04_client.py y el entrenamiento) ──────────
IMG_SIZE    = (128, 128)
CLASS_NAMES = ["1", "2", "3", "4", "5"]
MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/models/best_model.keras")

app = FastAPI(title="Hand Digit Inference Server")

# ── Cargar el modelo una sola vez al arrancar el servidor ─────────────────────
print(f"Cargando modelo desde {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Modelo no encontrado: {MODEL_PATH}. "
        "Asegúrate de que el pipeline de entrenamiento ha terminado."
    )
cnn_model = keras.models.load_model(MODEL_PATH)
# Llamada de calentamiento para que la primera petición real no tenga latencia extra
_ = cnn_model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
print("Modelo listo. Servidor escuchando en http://0.0.0.0:8000")


@app.get("/health")
def health():
    """Endpoint para comprobar que el servidor está activo."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen del esqueleto normalizado y devuelve la predicción.

    Body: multipart/form-data con campo 'file' (imagen PNG o JPEG).

    Response JSON:
        {
            "clase": "3",
            "dedos": 3,
            "confianza": 0.97,
            "probabilidades": {"1": 0.01, "2": 0.01, "3": 0.97, "4": 0.005, "5": 0.005}
        }
    """
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(
            status_code=415,
            detail="Solo se aceptan imágenes PNG o JPEG."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0   # normalizar a [0, 1]
        arr = arr[np.newaxis]                            # añadir dimensión de batch: (1, 128, 128, 3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar la imagen: {e}")

    probs  = cnn_model.predict(arr, verbose=0)[0]
    clase  = int(np.argmax(probs))
    confianza = float(probs[clase])

    return JSONResponse({
        "clase":           CLASS_NAMES[clase],
        "dedos":           clase + 1,
        "confianza":       round(confianza, 4),
        "probabilidades":  {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
    })
