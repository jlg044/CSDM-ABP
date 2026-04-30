import os
import pandas as pd
import shutil
from collections import Counter
from PIL import Image
import numpy as np

# Paths
csv_path = 'c:/Users/esthe/Desktop/Datos/Selu Izquierda/acciones_teclas.csv'
imagenes_path = 'c:/Users/esthe/Desktop/Imagenes Normalizadas/Selu Izquierda'
output_normalizado = 'c:/Users/esthe/Desktop/Dataset/Usuario16'

# Tamaño objetivo
tamaño_objetivo = (224, 224)

# Cargar CSV
df = pd.read_csv(csv_path, header=None, names=['timestamp', 'accion'])

# Crear estructura de acciones con (inicio, fin)
acciones = []
for i in range(0, len(df), 2):
    inicio = int(df.iloc[i]['timestamp'])
    fin = int(df.iloc[i+1]['timestamp'])
    accion = df.iloc[i]['accion']
    acciones.append({'accion': accion, 'inicio': inicio, 'fin': fin})

# Crear un diccionario para asignar cada timestamp a su acción correspondiente
timestamp_to_accion = {}
for accion in acciones:
    for ts in range(accion['inicio'], accion['fin'] + 1):
        timestamp_to_accion[ts] = accion['accion']

# Obtener todas las imágenes y ordenarlas
imagenes = sorted([img for img in os.listdir(imagenes_path) if img.endswith('.png')])

# Asociar cada imagen a su acción
imagenes_acciones = []
for img in imagenes:
    ts = int(img.split('.')[0])
    accion = timestamp_to_accion.get(ts, None)
    if accion:
        imagenes_acciones.append((img, accion))

# Crear carpetas por acción
for accion in set([acc for _, acc in imagenes_acciones]):
    accion_path = os.path.join(output_normalizado, accion)
    os.makedirs(accion_path, exist_ok=True)

# Agrupar de 10 en 10
contador_subcarpeta = {}
lote = []
for img, accion in imagenes_acciones:
    lote.append((img, accion))
    if len(lote) == 10:
        # Determinar acción predominante en el lote
        acciones_lote = [a for _, a in lote]
        accion_predominante = Counter(acciones_lote).most_common(1)[0][0]

        # Crear subcarpeta
        contador_subcarpeta.setdefault(accion_predominante, 0)
        contador_subcarpeta[accion_predominante] += 1
        subcarpeta = f"{accion_predominante}_{contador_subcarpeta[accion_predominante]}"
        subcarpeta_path = os.path.join(output_normalizado, accion_predominante, subcarpeta)
        os.makedirs(subcarpeta_path, exist_ok=True)

        # Procesar y guardar imágenes normalizadas
        for img_nombre, _ in lote:
            src = os.path.join(imagenes_path, img_nombre)
            dst = os.path.join(subcarpeta_path, img_nombre)

            imagen = Image.open(src).convert('RGB')
            imagen = imagen.resize(tamaño_objetivo)

            # Convertir a array, normalizar y volver a imagen para guardar
            imagen_array = np.array(imagen) / 255.0  # Normalizado a [0,1]
            imagen_array = (imagen_array * 255).astype(np.uint8)  # Reconvertir para guardar como png
            imagen_normalizada = Image.fromarray(imagen_array)

            imagen_normalizada.save(dst)

        lote = []  # Resetear lote

# Si quedan imágenes sueltas al final
if lote:
    acciones_lote = [a for _, a in lote]
    accion_predominante = Counter(acciones_lote).most_common(1)[0][0]

    contador_subcarpeta.setdefault(accion_predominante, 0)
    contador_subcarpeta[accion_predominante] += 1
    subcarpeta = f"{accion_predominante}_{contador_subcarpeta[accion_predominante]}"
    subcarpeta_path = os.path.join(output_normalizado, accion_predominante, subcarpeta)
    os.makedirs(subcarpeta_path, exist_ok=True)

    for img_nombre, _ in lote:
        src = os.path.join(imagenes_path, img_nombre)
        dst = os.path.join(subcarpeta_path, img_nombre)

        imagen = Image.open(src).convert('RGB')
        imagen = imagen.resize(tamaño_objetivo)

        imagen_array = np.array(imagen) / 255.0
        imagen_array = (imagen_array * 255).astype(np.uint8)
        imagen_normalizada = Image.fromarray(imagen_array)

        imagen_normalizada.save(dst)
