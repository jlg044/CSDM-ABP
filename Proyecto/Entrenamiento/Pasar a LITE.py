import tensorflow as tf

# Cargar el modelo .keras
model = tf.keras.models.load_model("c:/Users/esthe/Desktop/Modelo/modelo_acciones.keras")

# Crear el conversor
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir el modelo
tflite_model = converter.convert()

# Guardar el modelo .tflite
with open("c:/Users/esthe/Desktop/Modelo/modelo_acciones.tflite", "wb") as f:
    f.write(tflite_model)
