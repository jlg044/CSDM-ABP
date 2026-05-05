"""
02_train.py
-----------
Entrena la CNN sobre los splits generados por 01_preprocess.py.
Guarda el mejor modelo en /app/models/best_model.keras y el
historial de entrenamiento en /app/models/training_history.csv.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# Resolución a la que se redimensionan las imágenes antes de entrar a la red.
# 128×128 es suficiente para los esqueletos y reduce el coste computacional
# frente a 224×224, que empeoraba los resultados por overfitting.
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 16      # Lotes pequeños → gradientes más ruidosos pero mejor regularización
EPOCHS      = 1000    # Límite alto; EarlyStopping detiene antes si converge
CLASS_NAMES = ["1", "2", "3", "4", "5"]
SEED        = 42
SPLITS_DIR  = "/app/models"
MODEL_PATH  = os.path.join(SPLITS_DIR, "best_model.keras")

# Fijar semillas para reproducibilidad entre ejecuciones
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_cnn(input_shape=(128, 128, 3), num_classes=5):
    """
    CNN v4: diseño VGG-style con 4 bloques de doble convolución.

    Decisiones clave:
    - Doble Conv por bloque: compone características más complejas antes de hacer pooling.
    - BatchNormalization antes de ReLU: estabiliza activaciones y acelera convergencia.
    - SpatialDropout2D: desactiva canales enteros, más efectivo que Dropout en conv.
    - GlobalAveragePooling2D: evita el overfitting que causa Flatten en datasets pequeños.
    - L2 en Conv y Dense: penaliza pesos grandes para forzar soluciones más generales.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # ── Bloque 1: detecta bordes y líneas simples (32 filtros) ────────────
        layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),       # 128×128 → 64×64
        layers.SpatialDropout2D(0.1),     # dropout suave al inicio

        # ── Bloque 2: detecta esquinas y curvas (64 filtros) ──────────────────
        layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),       # 64×64 → 32×32
        layers.SpatialDropout2D(0.1),

        # ── Bloque 3: detecta segmentos de dedos (128 filtros) ────────────────
        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),       # 32×32 → 16×16
        layers.SpatialDropout2D(0.2),     # más dropout en capas profundas

        # ── Bloque 4: detecta la disposición global de dedos (256 filtros) ────
        layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.SpatialDropout2D(0.2),     # sin pooling: mantenemos resolución 16×16

        # ── Cabeza clasificadora ───────────────────────────────────────────────
        # GlobalAveragePooling promedia cada mapa 16×16 a un solo valor,
        # produciendo un vector de 256 en lugar de los 65.536 de un Flatten.
        # Esto elimina la mayor fuente de overfitting con datasets pequeños.
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.5),              # regularización fuerte antes de la salida
        layers.Dense(num_classes, activation="softmax"),  # probabilidad por clase
    ], name="HandDigit_CNN_v4")
    return model


def main():
    # Leer los splits generados por 01_preprocess.py.
    # dtype={"label": str} es imprescindible: PySpark guarda "1","2"... como
    # strings en el CSV pero pandas los leería como int, rompiendo el generador.
    df_train = pd.read_csv(os.path.join(SPLITS_DIR, "train_split.csv"), dtype={"label": str})
    df_val   = pd.read_csv(os.path.join(SPLITS_DIR, "val_split.csv"),   dtype={"label": str})
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # Parámetros comunes para ambos generadores
    FLOW_KW = dict(
        x_col="path", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASS_NAMES, seed=SEED,
    )

    # Generador de entrenamiento con augmentación:
    # Las transformaciones geométricas simulan variaciones reales de pose.
    # brightness_range fue descartado porque en esqueletos sobre fondo blanco
    # alterar el brillo distorsiona las líneas, perjudicando el entrenamiento.
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,        # normalizar píxeles al rango [0, 1]
        rotation_range=20,        # rotar hasta ±20°
        horizontal_flip=True,     # voltear horizontalmente (simetría de manos)
        zoom_range=0.15,          # zoom aleatorio ±15%
        width_shift_range=0.10,   # desplazamiento horizontal ±10%
        height_shift_range=0.10,  # desplazamiento vertical ±10%
    ).flow_from_dataframe(df_train, shuffle=True, **FLOW_KW)

    # Generador de validación: solo normalización, sin augmentación.
    # La validación debe reflejar las condiciones reales de inferencia.
    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
    ).flow_from_dataframe(df_val, shuffle=False, **FLOW_KW)

    # Calcular pesos de clase para compensar desbalances en el dataset.
    # "balanced" asigna más peso a las clases con menos muestras,
    # evitando que el modelo ignore las clases minoritarias.
    cw_arr = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2, 3, 4]), y=train_gen.classes,
    )
    class_weights = dict(enumerate(cw_arr))
    print("Class weights:", {CLASS_NAMES[k]: f"{v:.3f}" for k, v in class_weights.items()})

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        # Label smoothing 0.1: en lugar de one-hot duro [0,0,1,0,0],
        # usa [0.02, 0.02, 0.92, 0.02, 0.02]. Evita sobreconfianza y mejora generalización.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        # Detiene el entrenamiento si val_accuracy no mejora en 50 epochs seguidos.
        # patience=50 da margen para salir de mesetas sin parar prematuramente.
        # restore_best_weights recupera los pesos del mejor epoch al terminar.
        EarlyStopping(monitor="val_accuracy", patience=50, mode="max",
                      restore_best_weights=True, verbose=1),

        # Guarda el modelo solo cuando val_accuracy mejora.
        # Así el archivo guardado siempre contiene la mejor versión, no la última.
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),

        # Si val_loss no mejora en 5 epochs, divide el learning rate entre 2.
        # Permite ajustes más finos cuando el modelo se acerca a un óptimo.
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Guardar el historial completo para poder graficar las curvas en 03_evaluate.py
    pd.DataFrame(history.history).to_csv(
        os.path.join(SPLITS_DIR, "training_history.csv"), index=False,
    )
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
