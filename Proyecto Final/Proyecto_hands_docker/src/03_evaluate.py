"""
03_evaluate.py
--------------
Evalúa el modelo sobre el test set y guarda:
  - confusion_matrix.png
  - training_curves.png
  - metrics_summary.txt
en /app/models/
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
# Backend "Agg" renderiza gráficas en memoria sin necesitar pantalla.
# Obligatorio dentro del contenedor Docker, que no tiene entorno gráfico.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32          # batch mayor que en entrenamiento: solo hacemos forward pass
CLASS_NAMES = ["1", "2", "3", "4", "5"]
SPLITS_DIR  = "/app/models"
MODEL_PATH  = os.path.join(SPLITS_DIR, "best_model.keras")


def main():
    # dtype={"label": str} por el mismo motivo que en 02_train.py:
    # pandas leería las etiquetas como int y el generador fallaría
    df_test = pd.read_csv(os.path.join(SPLITS_DIR, "test_split.csv"), dtype={"label": str})
    print(f"Test set: {len(df_test)} imágenes")

    # Generador de test: sin augmentación, sin shuffle.
    # shuffle=False es crítico para que el orden de y_pred coincida con y_true.
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        df_test,
        x_col="path", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASS_NAMES,
        shuffle=False,
    )

    # Cargar el mejor modelo guardado por ModelCheckpoint durante el entrenamiento
    best_model = keras.models.load_model(MODEL_PATH)

    # Evaluación global: loss y accuracy sobre todo el test set
    test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
    print(f"\nTest Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # reset() rebobina el generador al inicio antes de predict(),
    # ya que evaluate() ya lo ha recorrido entero
    test_gen.reset()
    # predict() devuelve probabilidades por clase; argmax da el índice de la clase ganadora
    y_pred = np.argmax(best_model.predict(test_gen, verbose=0), axis=1)
    # test_gen.classes contiene las etiquetas reales en el mismo orden que las imágenes
    y_true = test_gen.classes

    label_names = [f"{c} dedo(s)" for c in CLASS_NAMES]
    report = classification_report(y_true, y_pred, target_names=label_names)
    print("\nInforme de clasificación:\n", report)

    # F1 por clase (average=None) y F1 macro (media de todas las clases sin ponderar)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_macro     = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ── Guardar métricas en texto ─────────────────────────────────────────────
    summary_path = os.path.join(SPLITS_DIR, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 45 + "\n")
        f.write("RESUMEN FINAL — TEST SET\n")
        f.write("=" * 45 + "\n")
        f.write(f"Accuracy global : {test_acc * 100:.2f}%\n")
        f.write(f"F1-score macro  : {f1_macro:.4f}\n")
        f.write("-" * 45 + "\n")
        for cls, f1 in zip(CLASS_NAMES, f1_per_class):
            f.write(f"  Clase {cls} dedo(s) — F1: {f1:.4f}\n")
        f.write("=" * 45 + "\n\n")
        f.write(report)
    print(f"Métricas guardadas en: {summary_path}")

    # ── Matriz de confusión ───────────────────────────────────────────────────
    cm      = confusion_matrix(y_true, y_pred)
    # Normalizar por fila: cada celda muestra la proporción sobre el total real de esa clase
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Dos subgráficas: conteos absolutos (izquierda) y proporciones (derecha)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax1)
    ax1.set_title("Confusión (conteos)")
    ax1.set_xlabel("Predicción"); ax1.set_ylabel("Real")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Confusión (proporción)")
    ax2.set_xlabel("Predicción"); ax2.set_ylabel("Real")

    plt.suptitle("Matrices de confusión — Test set", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(SPLITS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")

    # ── Curvas de entrenamiento ───────────────────────────────────────────────
    # Leer el CSV con el historial epoch a epoch guardado por 02_train.py
    hist_path = os.path.join(SPLITS_DIR, "training_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        ep   = range(1, len(hist) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Curva de accuracy: idealmente train y val convergen sin separarse mucho
        ax1.plot(ep, hist["accuracy"],     label="Train", linewidth=2)
        ax1.plot(ep, hist["val_accuracy"], label="Val",   linewidth=2, linestyle="--")
        ax1.set_title("Accuracy por epoch")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.legend(); ax1.grid(alpha=0.3)

        # Curva de loss: val_loss creciente con train_loss decreciente indica overfitting
        ax2.plot(ep, hist["loss"],     label="Train", linewidth=2)
        ax2.plot(ep, hist["val_loss"], label="Val",   linewidth=2, linestyle="--")
        ax2.set_title("Loss por epoch")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(SPLITS_DIR, "training_curves.png")
        plt.savefig(curves_path, dpi=120)
        plt.close()
        print(f"Curvas guardadas en: {curves_path}")

    print("\n=== Evaluación completada ===")


if __name__ == "__main__":
    main()
