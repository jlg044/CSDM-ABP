"""
01_preprocess.py
----------------
Usa PySpark para:
  - Listar todas las imágenes de /app/data
  - Extraer la etiqueta del nombre de archivo  (_1.0.png → 1)
  - Mostrar la distribución de clases
  - Dividir en train/val/test (70/15/15) de forma estratificada por clase
  - Guardar los splits como CSV en /app/models/
"""

import os
import re
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Directorio donde el contenedor Docker monta el dataset (solo lectura)
DATA_DIR   = "/app/data"
# Directorio de salida compartido con el host vía volumen Docker
OUTPUT_DIR = "/app/models"
# Semilla fija para que los splits sean reproducibles en cada ejecución
SEED       = 42

# URL del master de Spark. Si se define la variable de entorno SPARK_MASTER
# (como hace docker-compose con el cluster), se conecta al cluster real.
# Si no existe (ejecución local), usa todos los núcleos de la máquina.
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")


def main():
    # En modo cluster: el driver (este script) envía tareas a spark-master,
    # que las reparte entre spark-worker-1 y spark-worker-2.
    # En modo local: todo corre en el mismo proceso con local[*].
    print(f"Conectando a Spark master: {SPARK_MASTER}")
    spark = (
        SparkSession.builder
        .appName("HandDigit_DataSplit")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    # Reducir el nivel de log para que no ensucie la salida del pipeline
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Cargar rutas y etiquetas ──────────────────────────────────────────────
    # Los nombres de archivo siguen el patrón "..._1.0.png", "_2.0.png", etc.
    # La regex extrae el dígito entre el último guión bajo y ".0.png"
    records = []
    for fname in os.listdir(DATA_DIR):
        m = re.search(r"_(\d)\.0\.png$", fname)
        if m:
            # Guardamos la ruta absoluta y la etiqueta como string ("1".."5")
            records.append((os.path.join(DATA_DIR, fname), m.group(1)))

    if not records:
        raise RuntimeError(f"No se encontraron imágenes en {DATA_DIR}")

    # Crear DataFrame de Spark con dos columnas: ruta del archivo y etiqueta
    df = spark.createDataFrame(records, ["path", "label"])
    total = df.count()
    print(f"\nTotal imágenes: {total}")

    # Mostrar cuántas imágenes hay por clase para verificar el balance
    print("\nDistribución de clases:")
    df.groupBy("label").count().orderBy("label").show()

    # ── Split estratificado por clase (70 / 15 / 15) ─────────────────────────
    # Hacemos el split clase a clase para garantizar que cada clase mantenga
    # su proporción en los tres conjuntos (estratificado).
    # Si hiciéramos un split global, clases con pocas muestras podrían quedar
    # desproporcionadas por azar.
    train_parts, temp_parts = [], []
    for lbl in ["1", "2", "3", "4", "5"]:
        subset = df.filter(F.col("label") == lbl)
        # 70% para entrenamiento, 30% para repartir entre val y test
        tr, te = subset.randomSplit([0.70, 0.30], seed=SEED)
        train_parts.append(tr)
        temp_parts.append(te)

    # Reunir los fragmentos de todas las clases en un único DataFrame
    train = reduce(lambda a, b: a.union(b), train_parts)
    temp  = reduce(lambda a, b: a.union(b), temp_parts)

    # Dividir el 30% restante a la mitad: 15% validación, 15% test
    val_parts, test_parts = [], []
    for lbl in ["1", "2", "3", "4", "5"]:
        subset = temp.filter(F.col("label") == lbl)
        v, t = subset.randomSplit([0.50, 0.50], seed=SEED)
        val_parts.append(v)
        test_parts.append(t)

    val  = reduce(lambda a, b: a.union(b), val_parts)
    test = reduce(lambda a, b: a.union(b), test_parts)

    # ── Guardar como CSV ──────────────────────────────────────────────────────
    # toPandas() trae los datos al driver para poder usar to_csv().
    # El CSV resultante tiene columnas "path" y "label" que leerá 02_train.py.
    for split_df, name in [(train, "train"), (val, "val"), (test, "test")]:
        out_path = os.path.join(OUTPUT_DIR, f"{name}_split.csv")
        split_df.toPandas().to_csv(out_path, index=False)
        print(f"{name:5s}: {split_df.count():4d} imágenes → {out_path}")

    print("\nPreprocesado completado.")
    spark.stop()


if __name__ == "__main__":
    main()
