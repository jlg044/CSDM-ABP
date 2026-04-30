import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    TimeDistributed,
    LSTM,
    Dense,
    Flatten,
    GRU
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
dataset_path = 'Entrenamiento_derecha/'

# Parámetros
img_height = 224
img_width = 224
imgs_per_sequence = 10
IMG_SHAPE = (imgs_per_sequence, img_height, img_width, 3)
num_classes = -1

# ======================================================
# UTILIDADES DE CARGA
# ======================================================
def load_sequence(sequence_path, img_shape):
    """
    Carga una secuencia de imágenes desde un directorio, las normaliza y las almacena en un array.

    Args:
        sequence_path (str): Ruta al directorio que contiene las imágenes.
        img_shape (tuple): Forma deseada del tensor de salida (frames, height, width, channels).

    Returns:
        np.ndarray: Secuencia de imágenes con forma `img_shape` y valores normalizados [0, 1].
    """
    num_frames, height, width, channels = img_shape
    sequence_frames = np.zeros(shape=img_shape, dtype=np.float32)

    files = sorted(os.listdir(sequence_path))[:num_frames]

    for i, file_name in enumerate(files):
        img_path = os.path.join(sequence_path, file_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

            if img is None:
                # Imagen no pudo ser leída (formato incorrecto o dañada)
                continue

            img = cv2.resize(img, (width, height))  # Ajustar tamaño
            img = img.astype(np.float32) / 255.0    # Normalización
            sequence_frames[i] = img
        else:
            # Imagen no encontrada: ya está inicializado con ceros, así que no se necesita nada
            pass

    return sequence_frames

def build_sequence_list(base_dir, subset):
    all_sequences = []
    users = os.listdir(base_dir)

    class_names = sorted([d for d in os.listdir(os.path.join(base_dir, users[0], subset)) 
                          if os.path.isdir(os.path.join(base_dir, users[0], subset, d)) and "check" not in d])
    for class_idx, class_name in enumerate(class_names):
        for user in users:
            class_dir = os.path.join(base_dir, user, subset, class_name)
            for seq in os.listdir(class_dir):
                seq_path = os.path.join(class_dir, seq)
                if os.path.isdir(seq_path) and "checkpoint" not in seq_path:
                    label = np.zeros(len(class_names), dtype=np.float32)
                    label[class_idx] = 1.0
                    all_sequences.append((seq_path, label))
    return all_sequences, len(class_names)

# ======================================================
# PIPELINE DE PARSE Y TF.DATA
# ======================================================
def parse_indices(indices, all_sequences, img_shape):
    images, labels = zip(*[
        (load_sequence(all_sequences[i][0], img_shape), all_sequences[i][1]) for i in indices
    ])
    return np.stack(images), np.stack(labels)

def get_tf_dataset(all_sequences, img_shape, num_classes, batch_size, shuffle=True):
    def tf_parse_fn(batch_indices):
        imgs, lbls = tf.numpy_function(
            lambda idxs: parse_indices(idxs, all_sequences, img_shape),
            [batch_indices],
            [tf.float32, tf.float32]
        )
        imgs.set_shape((None,) + img_shape)
        lbls.set_shape((None, num_classes))
        return imgs, lbls

    indices = np.arange(len(all_sequences))
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(tf_parse_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds

# ======================================================
# DIVISIÓN Y CREACIÓN DE DATASETS
# ======================================================
def prepare_datasets(base_dir, subset, img_shape, batch_size=8, split_validation=False):
    all_sequences, num_classes = build_sequence_list(base_dir, subset)
    np.random.seed(42)
    np.random.shuffle(all_sequences)

    if split_validation and subset == "train":
        split = int(0.8 * len(all_sequences))
        train_seqs = all_sequences[:split]
        val_seqs = all_sequences[split:]
        train_ds = get_tf_dataset(train_seqs, img_shape, num_classes, batch_size)
        val_ds = get_tf_dataset(val_seqs, img_shape, num_classes, batch_size, shuffle=False)
        return train_ds, val_ds, num_classes

    ds = get_tf_dataset(all_sequences, img_shape, num_classes, batch_size)
    return ds, None, num_classes

# ======================================================
# CARGA DE DATOS
# ======================================================
IMG_SHAPE = (10, 224, 224, 3)
train_ds, val_ds, num_classes = prepare_datasets(dataset_path, "train", IMG_SHAPE, batch_size=4, split_validation=True)
test_ds, _, _ = prepare_datasets(dataset_path, "test", IMG_SHAPE, batch_size=4, split_validation=False)

# ======================================================
# DEFINICIÓN DEL MODELO
# ======================================================
def model_v1():
    sequence_input = Input(shape=(10,224,224,3))
    cnn = TimeDistributed(Conv2D(4, (3,3), activation='relu'))(sequence_input)
    cnn = TimeDistributed(MaxPooling2D((2,2)))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.15))(cnn)

    cnn = TimeDistributed(Conv2D(8, (3,3), activation='relu'))(cnn)
    cnn = TimeDistributed(MaxPooling2D((2,2)))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.15))(cnn)

    cnn = TimeDistributed(Conv2D(8, (3,3), activation='relu'))(cnn)
    cnn = TimeDistributed(MaxPooling2D((4,4)))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.15))(cnn)

    cnn = TimeDistributed(Conv2D(16, (5,5), activation='relu'))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.15))(cnn)

    cnn = TimeDistributed(Conv2D(16, (5,5), activation='relu'))(cnn)
    cnn = TimeDistributed(MaxPooling2D((2,2)))(cnn)
    cnn = TimeDistributed(BatchNormalization())(cnn)
    cnn = TimeDistributed(Dropout(0.15))(cnn)
    
    cnn = TimeDistributed(GlobalAveragePooling2D())(cnn)

    gru = GRU(10,unroll=True)(cnn)
    gru = Dropout(0.15)(gru)
    #flat = Flatten()(cnn)
    dense = Dense(10, activation='relu')(gru)
    dense = Dropout(0.15)(dense)

    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=sequence_input, outputs=output)
    return model

# ======================================================
# COMPILACIÓN Y CALLBACKS
# ======================================================
model = model_v1()
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='mejor_modelo.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print(model.summary())
# ======================================================
# ENTRENAMIENTO
# ======================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1000,
    callbacks=[early_stop, checkpoint]
)

# Guardar historial
np.save('historial_entrenamiento.npy', history.history)

# ======================================================
# EVALUACIÓN Y GUARDADO FINAL
# ======================================================
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
model.save('modelo_acciones.keras')
print("Modelo guardado como 'modelo_acciones.keras'")

# ======================================================
# GRÁFICOS DE ENTRENAMIENTO Y VALIDACIÓN
# ======================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Evolución de la precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Evolución de la pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
