# P7 — Clasificación de Dedos con CNN (Docker + PySpark + TensorFlow)

Proyecto de clasificación de imágenes que detecta cuántos dedos se muestran (1 a 5) usando una CNN entrenada desde cero sobre esqueletos de mano generados con MediaPipe.

---

## Tecnologías utilizadas

| Tecnología | Versión | Rol en el proyecto |
|---|---|---|
| Python | 3.11 | Lenguaje base dentro del contenedor |
| TensorFlow / Keras | 2.15.1 | Definición y entrenamiento de la CNN |
| PySpark | 3.5.1 | Carga distribuida del dataset y split estratificado |
| OpenCV (headless) | 4.9.0 | Lectura de imágenes y procesamiento de vídeo |
| MediaPipe | ≥0.10.14 | Detección de landmarks de mano en tiempo real |
| scikit-learn | ≥1.3 | Pesos de clase, métricas F1 y matriz de confusión |
| Matplotlib / Seaborn | ≥3.7 / ≥0.13 | Gráficas de entrenamiento y matrices de confusión |
| FastAPI + Uvicorn | ≥0.110 / ≥0.29 | Servidor REST de inferencia en tiempo real (dentro de Docker) |
| Docker + Compose | — | Contenedorización del pipeline de entrenamiento e inferencia |
| Java (OpenJDK) | headless | Requerido internamente por PySpark |

---

## Arquitectura del pipeline

```
Imagenes Normalizadas/          ← dataset en la carpeta padre
        │
        ▼
01_preprocess.py  (PySpark)
  · Escanea los PNG y extrae etiqueta del nombre de archivo
  · Distribución de clases por consola
  · Split estratificado 70 / 15 / 15 → CSV
        │
        ▼
02_train.py  (TensorFlow / Keras)
  · Carga splits con ImageDataGenerator + flow_from_dataframe
  · Augmentación: rotación, flip, zoom, desplazamiento
  · Pesos de clase para equilibrar el dataset
  · Entrena CNN v4 (hasta 1 000 epochs con EarlyStopping)
  · Guarda el mejor modelo en /app/models/best_model.keras
        │
        ▼
03_evaluate.py  (sklearn + matplotlib)
  · Evalúa sobre el test set
  · Genera confusion_matrix.png y training_curves.png
  · Escribe metrics_summary.txt con accuracy y F1 por clase
        │
        ▼  (dentro de Docker)
04_server.py  (FastAPI + Keras)
  · Carga el modelo entrenado una sola vez al arrancar
  · Expone POST /predict: recibe imagen PNG del esqueleto → devuelve JSON con la predicción
  · Expone GET /health para comprobar disponibilidad
  · Escucha en http://localhost:8000
        │
        ▼  (fuera de Docker, localmente)
04_client.py  (OpenCV + MediaPipe + requests)
  · Abre la cámara web
  · Detecta la mano en cada frame con MediaPipe HandLandmarker
  · Normaliza el esqueleto igual que en el preprocesado
  · Envía el esqueleto al servidor Docker vía HTTP
  · Muestra la predicción en pantalla en tiempo real
```

---

## Arquitectura de la CNN (v4)

La red se entrena **desde cero** sin transfer learning, siguiendo un diseño VGG-style con cuatro bloques de doble convolución:

```
Input (128 × 128 × 3)
│
├─ Bloque 1: Conv2D(32)×2 → BN → ReLU → MaxPool(2) → SpatialDropout(0.1)
├─ Bloque 2: Conv2D(64)×2 → BN → ReLU → MaxPool(2) → SpatialDropout(0.1)
├─ Bloque 3: Conv2D(128)×2 → BN → ReLU → MaxPool(2) → SpatialDropout(0.2)
├─ Bloque 4: Conv2D(256)×2 → BN → ReLU             → SpatialDropout(0.2)
│
├─ GlobalAveragePooling2D   ← evita overfitting frente a Flatten
├─ Dense(256, relu) + L2
├─ Dropout(0.5)
└─ Dense(5, softmax)
```

**Decisiones de diseño:**

- **GlobalAveragePooling2D** en lugar de Flatten: reduce drásticamente los parámetros y el riesgo de sobreajuste en datasets pequeños.
- **BatchNormalization** tras cada Conv: estabiliza y acelera el entrenamiento.
- **SpatialDropout2D** en mapas de características: regularización espacial más efectiva que el Dropout estándar para imágenes.
- **Label smoothing (0.1)**: evita que el modelo sea demasiado confiado y mejora la generalización.
- **Pesos de clase**: compensa el desbalance entre clases sin descartar muestras.
- **LR = 1 × 10⁻⁴** con `ReduceLROnPlateau` (factor 0.5, patience 5): bajada suave cuando el val_loss se estanca.
- **EarlyStopping patience = 50**: margen amplio para que el modelo salga de mesetas sin parar prematuramente.
- **Batch size = 16**: compromiso entre estabilidad del gradiente y uso de memoria en el contenedor.

---

## Estructura de archivos

```
Proyecto_hands_docker/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── src/
│   ├── 01_preprocess.py   ← PySpark: split del dataset
│   ├── 02_train.py        ← Keras: entrenamiento de la CNN
│   ├── 03_evaluate.py     ← métricas y gráficas
│   ├── 04_server.py       ← FastAPI: servidor de inferencia (Docker)
│   └── 04_client.py       ← cliente con cámara (local)
└── models/                ← generado al ejecutar el pipeline
    ├── best_model.keras
    ├── train_split.csv
    ├── val_split.csv
    ├── test_split.csv
    ├── training_history.csv
    ├── confusion_matrix.png
    ├── training_curves.png
    └── metrics_summary.txt
```

---

## Requisitos previos

- **Docker Desktop** instalado y en ejecución
- El dataset en `../Imagenes Normalizadas/` (carpeta hermana de `Proyecto_hands_docker/`)
- Para la inferencia en tiempo real: Python local con las librerías del cliente instaladas (ver Paso 3)

---

## Paso 1 — Entrenar con Docker (pipeline completo)

Desde dentro de `Proyecto_hands_docker/`:

```bash
cd Proyecto_hands_docker
docker compose up --build
```

Esto construye la imagen y lanza **4 contenedores en orden**:

| Contenedor | Qué hace |
|---|---|
| `spark-master` | Nodo maestro del cluster Spark |
| `spark-worker-1` | Worker 1 |
| `spark-worker-2` | Worker 2 |
| `cnn-trainer` | Ejecuta `01_preprocess.py` → `02_train.py` → `03_evaluate.py` |
| `inference-server` | Arranca cuando `best_model.keras` existe y queda escuchando en el puerto 8000 |

Al terminar el entrenamiento, todos los resultados quedan en `Proyecto_hands_docker/models/`.

**Para parar y limpiar:**

```bash
docker compose down               # para los contenedores
docker compose down --rmi local   # elimina también la imagen
```

---

## Paso 2 — Ver los resultados del entrenamiento

```bash
type Proyecto_hands_docker\models\metrics_summary.txt
```

O abre directamente en el explorador de archivos:
- `models/training_curves.png`
- `models/confusion_matrix.png`

---

## Paso 3 — Inferencia en tiempo real con la cámara

El servidor de inferencia ya está corriendo en Docker (Paso 1). Solo necesitas lanzar el cliente localmente.

**Instalación de dependencias locales (una sola vez):**

```bash
pip install opencv-python mediapipe numpy requests pillow
```

**Ejecutar el cliente:**

```bash
cd Proyecto_hands_docker
python src/04_client.py
```

Si el servidor está en una URL distinta:

```bash
python src/04_client.py --server http://localhost:8000
```

El cliente espera automáticamente a que el servidor responda antes de abrir la cámara.
La primera vez descarga automáticamente `hand_landmarker.task` (~25 MB).
Muestra tu mano a la cámara y pulsa **Q** para salir.

---

## Ver los logs en tiempo real

```bash
# Solo el entrenamiento
docker logs -f cnn-hand-trainer

# Solo el servidor de inferencia
docker logs -f inference-server

# Todo a la vez
docker compose logs -f
```

---

## Solución de problemas

**El dataset no se encuentra:**
```bash
ls "../Imagenes Normalizadas/" | head -5
```
Asegúrate de que la carpeta `Imagenes Normalizadas` está en el directorio padre de `Proyecto_hands_docker/`.

**Error de memoria durante el entrenamiento:**
Edita `docker-compose.yml` y añade bajo `cnn-trainer`:
```yaml
    deploy:
      resources:
        limits:
          memory: 4g
```

**El modelo ya existe y no quiero sobreescribirlo:**
```bash
cp models/best_model.keras models/best_model_backup.keras
```

**Cámara no se abre en tiempo real:**
Comprueba que ninguna otra aplicación está usando la cámara y que Python tiene permisos de acceso (en Windows: Configuración → Privacidad → Cámara).

**El servidor de inferencia no responde:**
```bash
docker logs inference-server
```
Asegúrate de que el entrenamiento ha terminado y `models/best_model.keras` existe.
El contenedor espera automáticamente a ese archivo antes de arrancar Uvicorn.
