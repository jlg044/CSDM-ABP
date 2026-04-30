import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, InputLayer

# Parámetros
sequence_length = 10
img_height, img_width = 224, 224
batch_size = 10
epochs = 10
dataset_path = 'C:/Users/esthe/Desktop/Entrenamiento/train'
test_path = 'C:/Users/esthe/Desktop/Entrenamiento/test'

# =================================================================
# FUNCIONES PARA CARGAR SECUENCIAS
# =================================================================
def load_sequences_from_nested_directory(base_dir, sequence_length=10):
    X, y = [], []
    label_names = set()

    for usuario in sorted(os.listdir(base_dir)):
        usuario_path = os.path.join(base_dir, usuario)
        if not os.path.isdir(usuario_path):
            continue

        for accion in sorted(os.listdir(usuario_path)):
            accion_path = os.path.join(usuario_path, accion)
            if not os.path.isdir(accion_path):
                continue
            label_names.add(accion)

    label_names = sorted(list(label_names))
    label_map = {label: idx for idx, label in enumerate(label_names)}

    for usuario in sorted(os.listdir(base_dir)):
        usuario_path = os.path.join(base_dir, usuario)
        if not os.path.isdir(usuario_path):
            continue

        for accion in sorted(os.listdir(usuario_path)):
            accion_path = os.path.join(usuario_path, accion)
            if not os.path.isdir(accion_path):
                continue

            for secuencia in sorted(os.listdir(accion_path)):
                secuencia_path = os.path.join(accion_path, secuencia)
                if not os.path.isdir(secuencia_path):
                    continue

                frames = sorted(os.listdir(secuencia_path))
                if len(frames) < sequence_length:
                    continue  # Ignorar si hay menos de 10 imágenes

                sequence = []
                for frame_name in frames[:sequence_length]:
                    frame_path = os.path.join(secuencia_path, frame_name)
                    img = cv2.imread(frame_path)
                    img = cv2.resize(img, (img_width, img_height))
                    img = img.astype(np.float32) / 255.0
                    sequence.append(img)

                X.append(sequence)
                y.append(label_map[accion])

    return np.array(X), to_categorical(y, num_classes=len(label_map)), len(label_map)


# =================================================================
# CARGA DE DATOS
# =================================================================
print("Cargando datos de entrenamiento...")
X_train, y_train, num_classes = load_sequences_from_nested_directory(dataset_path)

print("Cargando datos de test...")
X_test, y_test, _ = load_sequences_from_nested_directory(test_path)

# =================================================================
# DEFINICIÓN DEL MODELO
# =================================================================
model = Sequential([
    InputLayer(input_shape=(sequence_length, img_height, img_width, 3)),

    TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),

    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),

    TimeDistributed(Flatten()),

    LSTM(64),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =================================================================
# ENTRENAMIENTO
# =================================================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size
)

# =================================================================
# EVALUACIÓN
# =================================================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# =================================================================
# GUARDADO
# =================================================================
model.save('modelo_acciones_10_frames.keras')
print("Modelo guardado como 'modelo_acciones_10_frames.keras'")
