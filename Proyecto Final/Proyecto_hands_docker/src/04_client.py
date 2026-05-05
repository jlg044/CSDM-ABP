"""
04_client.py
------------
Cliente de inferencia en tiempo real (corre LOCALMENTE, fuera de Docker).

Flujo:
  1. Captura frames de la cámara con OpenCV
  2. Detecta la mano con MediaPipe HandLandmarker
  3. Normaliza los landmarks al mismo formato que el entrenamiento
  4. Envía el esqueleto PNG al servidor Docker vía POST /predict
  5. Muestra la predicción (clase + confianza) sobre el vídeo en pantalla

Uso:
    python src/04_client.py
    python src/04_client.py --server http://localhost:8000

Dependencias locales (instalar una vez):
    pip install opencv-python mediapipe numpy requests pillow
"""

import argparse
import io
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import requests
from PIL import Image

# ── Constantes (deben coincidir con el servidor y el dataset de entrenamiento) ─
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]
NORM_SIZE      = 224
CENTER_PIXEL   = NORM_SIZE // 2
HAND_BOX_SIZE  = 160
IMG_SIZE       = (128, 128)
CLASS_NAMES    = ["1", "2", "3", "4", "5"]
LANDMARK_MODEL = "hand_landmarker.task"
SERVER_URL     = "http://localhost:8000"


def normalizar_mano(landmarks_list):
    """
    Replica exactamente la normalización de Normalizacion.py:
      1. Trasladar al origen (muñeca en 0,0)
      2. Escalar a HAND_BOX_SIZE
      3. Centrar en el lienzo NORM_SIZE×NORM_SIZE
      4. Dibujar esqueleto (huesos rojos, articulaciones azules, fondo blanco)
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks_list])
    pts = pts - pts[0]

    min_xy   = pts.min(axis=0)
    max_xy   = pts.max(axis=0)
    max_side = max(max_xy - min_xy)
    escala   = HAND_BOX_SIZE / max_side if max_side != 0 else 1

    centro_box = (min_xy + max_xy) / 2.0
    offset_x   = CENTER_PIXEL - centro_box[0] * escala
    offset_y   = CENTER_PIXEL - centro_box[1] * escala

    px = [(int(x * escala + offset_x), int(y * escala + offset_y))
          for (x, y) in pts.tolist()]

    canvas = np.ones((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8) * 255
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], (0, 0, 255), 2)
    for p in px:
        cv2.circle(canvas, p, 3, (255, 0, 0), -1)
    return canvas


def canvas_to_bytes(canvas_bgr):
    """Convierte un array BGR de OpenCV a bytes PNG listos para enviar por HTTP."""
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(canvas_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        default=SERVER_URL,
        help="URL base del servidor de inferencia (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    predict_url = f"{args.server.rstrip('/')}/predict"

    # Comprobar que el servidor está disponible antes de abrir la cámara
    print(f"Conectando al servidor: {args.server}")
    for intento in range(10):
        try:
            r = requests.get(f"{args.server}/health", timeout=2)
            if r.status_code == 200:
                print("Servidor listo.")
                break
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Esperando al servidor... ({intento + 1}/10)")
        time.sleep(2)
    else:
        print("ERROR: No se pudo conectar al servidor. ¿Está corriendo Docker?")
        return

    # Descargar el modelo de landmarks si no existe localmente
    if not os.path.exists(LANDMARK_MODEL):
        print("Descargando hand_landmarker.task (~25 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            LANDMARK_MODEL,
        )
        print("Descargado.")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(0)
    print("Cámara abierta — muestra tu mano. Pulsa Q para salir.")

    # Estado de la última predicción (se actualiza de forma asíncrona)
    last_label  = "Sin mano detectada"
    last_color  = (120, 120, 120)
    last_canvas = None

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            if result.hand_landmarks:
                landmarks  = result.hand_landmarks[0]
                canvas     = normalizar_mano(landmarks)
                last_canvas = canvas

                # Enviar el esqueleto al servidor Docker
                try:
                    png_bytes = canvas_to_bytes(canvas)
                    resp = requests.post(
                        predict_url,
                        files={"file": ("skeleton.png", png_bytes, "image/png")},
                        timeout=1.0,
                    )
                    if resp.status_code == 200:
                        data       = resp.json()
                        confianza  = data["confianza"] * 100
                        last_label = f"{data['clase']} dedo(s)   {confianza:.1f}%"
                        last_color = (0, 210, 0)
                    else:
                        last_label = f"Error servidor: {resp.status_code}"
                        last_color = (0, 0, 200)
                except requests.exceptions.Timeout:
                    last_label = "Servidor lento (timeout)"
                    last_color = (0, 140, 255)
                except requests.exceptions.ConnectionError:
                    last_label = "Sin conexión al servidor"
                    last_color = (0, 0, 200)
            else:
                last_label  = "Sin mano detectada"
                last_color  = (120, 120, 120)
                last_canvas = None

            # Mostrar el esqueleto en la esquina superior izquierda
            if last_canvas is not None:
                thumb = cv2.resize(last_canvas, (150, 150))
                frame[10:160, 10:160] = thumb
                cv2.rectangle(frame, (10, 10), (160, 160), (0, 0, 0), 1)

            # Barra de estado inferior con la predicción
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 55), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, last_label, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, last_color, 2, cv2.LINE_AA)

            cv2.imshow("Hand Digit Client  |  Q para salir", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada.")


if __name__ == "__main__":
    main()
