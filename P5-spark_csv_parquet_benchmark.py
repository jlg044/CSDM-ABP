import os
import time
import matplotlib
# Usamos Agg para que no de problemas al guardar la grafica en el servidor de Docker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    StringType,
)

# Definimos las rutas para no tener que escribirlas cada vez
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "student_productivity_distraction_dataset_20000.csv")
OUT_DIR = os.path.join(BASE_DIR, "output", "p5_formal_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

# Definimos el esquema a mano para que Spark no tenga que adivinarlo (va mas rapido)
schema = StructType([
    StructField("student_id", IntegerType(), True),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("study_hours_per_day", FloatType(), True),
    StructField("sleep_hours", FloatType(), True),
    StructField("phone_usage_hours", FloatType(), True),
    StructField("social_media_hours", FloatType(), True),
    StructField("youtube_hours", FloatType(), True),
    StructField("gaming_hours", FloatType(), True),
    StructField("breaks_per_day", IntegerType(), True),
    StructField("coffee_intake_mg", IntegerType(), True),
    StructField("exercise_minutes", IntegerType(), True),
    StructField("assignments_completed", IntegerType(), True),
    StructField("attendance_percentage", FloatType(), True),
    StructField("stress_level", IntegerType(), True),
    StructField("focus_score", IntegerType(), True),
    StructField("final_grade", FloatType(), True),
    StructField("productivity_score", FloatType(), True),
])

# Pequeña funcion para medir tiempos sin repetir codigo
def timed(fn):
    start = time.perf_counter()
    value = fn()
    end = time.perf_counter()
    return value, end - start

def main():
    # Creamos la sesion de Spark
    spark = (
        SparkSession.builder
        .appName("Benchmark_CSV_vs_Parquet_Grupo")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    results = []

    # 1. Empezamos con el CSV original para tener una base
    print("--- Cargando y probando CSV ---")
    df_csv = spark.read.csv(CSV_PATH, header=True, schema=schema)
    def query_csv():
        # Filtramos por mujeres y sumamos su productividad
        return df_csv.filter(F.col("gender") == "Female").agg(F.sum("productivity_score")).collect()[0]
    _, t_csv = timed(query_csv)
    results.append({"name": "CSV (Original)", "time": t_csv})

    # 2. Convertimos a Parquet pelado (sin compresion nada)
    print("--- Probando Parquet sin compresion ---")
    path_plain = os.path.join(OUT_DIR, "plain.parquet")
    df_csv.write.mode("overwrite").option("compression", "uncompressed").parquet(path_plain)
    df_plain = spark.read.parquet(path_plain)
    _, t_plain = timed(lambda: df_plain.filter(F.col("gender") == "Female").agg(F.sum("productivity_score")).collect()[0])
    results.append({"name": "Parquet (Sin Compresion)", "time": t_plain})

    # 3. Ahora con Snappy, que es lo que se recomienda normalmente
    print("--- Probando Snappy ---")
    path_snappy = os.path.join(OUT_DIR, "snappy.parquet")
    df_csv.write.mode("overwrite").option("compression", "snappy").parquet(path_snappy)
    df_snappy = spark.read.parquet(path_snappy)
    _, t_snappy = timed(lambda: df_snappy.filter(F.col("gender") == "Female").agg(F.sum("productivity_score")).collect()[0])
    results.append({"name": "Snappy (Comprimido)", "time": t_snappy})

    # 4. Probamos el Pushdown con baja cardinalidad (Genero)
    print("--- Analizando Pushdown (Genero) ---")
    spark.conf.set("spark.sql.parquet.filterPushdown", "true")
    _, t_low_card = timed(lambda: df_snappy.filter(F.col("gender") == "Female").agg(F.sum("productivity_score")).collect()[0])
    results.append({"name": "Pushdown (Baja Card.)", "time": t_low_card})

    # 5. Y ahora con mucha cardinalidad (IDs de estudiantes)
    print("--- Analizando Pushdown (IDs) ---")
    _, t_high_card = timed(lambda: df_snappy.filter(F.col("student_id") == 5000).agg(F.sum("productivity_score")).collect()[0])
    results.append({"name": "Pushdown (Alta Card.)", "time": t_high_card})

    # 6. Particionado por genero (en carpetas separadas)
    print("--- Haciendo el particionado ---")
    path_part = os.path.join(OUT_DIR, "partitioned")
    df_csv.write.mode("overwrite").partitionBy("gender").parquet(path_part)
    df_part = spark.read.parquet(path_part)
    # Aqui se deberia notar mucha diferencia por el Partition Pruning
    _, t_part = timed(lambda: df_part.filter(F.col("gender") == "Female").agg(F.sum("productivity_score")).collect()[0])
    results.append({"name": "Particionado (Genero)", "time": t_part})

    # 7. Schema Merging: juntamos dos trozos con esquemas algo distintos
    print("--- Haciendo Schema Merging ---")
    path_a = os.path.join(OUT_DIR, "merge_a")
    path_b = os.path.join(OUT_DIR, "merge_b")
    df_csv.limit(10000).write.mode("overwrite").parquet(path_a)
    # A un trozo le añadimos una columna extra para forzar el merge
    df_b_raw = df_csv.filter(F.col("student_id") > 10000).withColumn("extra_col", F.lit(1))
    df_b_raw.write.mode("overwrite").parquet(path_b)
    _, t_merge = timed(lambda: spark.read.option("mergeSchema", "true").parquet(path_a, path_b).count())
    results.append({"name": "Schema Merging (Union)", "time": t_merge})

    # 8. Bucketing (JOIN): esto es clave para optimizar cruces de tablas
    print("--- Bucketing y Join ---")
    import shutil
    # Borramos la carpeta primero por si acaso, que si no Spark da fallo de LocationExists
    warehouse_path = os.path.join(BASE_DIR, "spark-warehouse", "t_bucket")
    if os.path.exists(warehouse_path):
        shutil.rmtree(warehouse_path)

    spark.sql("DROP TABLE IF EXISTS t_bucket")
    df_csv.write.mode("overwrite").bucketBy(8, "student_id").sortBy("student_id").saveAsTable("t_bucket")
    def run_join():
        t1 = spark.table("t_bucket")
        t2 = spark.table("t_bucket")
        return t1.join(t2, "student_id").count()
    _, t_bucket = timed(run_join)
    results.append({"name": "Bucketing (Join)", "time": t_bucket})

    # --- Generamos la grafica para la presentacion ---
    print("\nCreando la grafica final...")
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]

    plt.figure(figsize=(15, 9), facecolor='#1a1a2e')
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    
    # Usamos colores variados para que se diferencien bien las pruebas
    colors = ['#ff4d4d', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#9b59b6', '#1abc9c', '#3498db']

    bars = plt.bar(names, times, color=colors, edgecolor='white', linewidth=1)
    
    plt.title("COMPARATIVA DE RENDIMIENTO: NUESTRAS PRUEBAS EN SPARK", fontsize=18, fontweight='bold', color='white', pad=35)
    plt.suptitle("Benchmark de tiempos entre CSV, Parquet y optimizaciones (20k registros)", fontsize=12, color='#cccccc', y=0.92)
    plt.ylabel("Segundos", color='white', fontsize=12)
    
    # Rotamos las etiquetas para que no se pisen entre ellas
    plt.xticks(rotation=30, ha='right', color='white', fontsize=11, fontweight='bold')
    plt.yticks(color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.1, color='white')

    # Ponemos el tiempo encima de cada barra para que se lea mejor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + (max(times)*0.01),
                 f'{height:.4f}s', ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color='white')

    # Dejamos espacio abajo para los nombres
    plt.subplots_adjust(bottom=0.3, top=0.85) 
    
    # Guardamos la imagen final para el reporte
    save_path = os.path.join(OUT_DIR, "P5-grafica_final.png")
    plt.savefig(save_path, dpi=200, facecolor='#1a1a2e', bbox_inches='tight')
    
    print(f"\nProceso terminado.")
    print(f"La grafica la tienes aqui: {save_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()
