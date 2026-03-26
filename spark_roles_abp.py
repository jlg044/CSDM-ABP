import time
import csv
import random
from pathlib import Path
from py4j.protocol import Py4JJavaError
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import functions as F

# Abrimos la sesión de Spark. 
# local[*] para que el PC use todos los cores que tenga libres.
spark = SparkSession.builder \
    .appName("RetoSpark_ABP_Master_Final") \
    .master("local[*]") \
    .getOrCreate()

print("\n" + "="*60)
print("  🚀 ARRANCAMOS EL RETO SPARK ABP - EQUIPO DE DATOS")
print("="*60)

# -------------------------------------------------------------
# P2 - Arquitecto de Datos: El esquema y la carga de datos
# -------------------------------------------------------------

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
    StructField("productivity_score", FloatType(), True)
])

csv_filename = "student_productivity_distraction_dataset_20000.csv"


def generar_dataset_sintetico(csv_file: Path, n_rows: int = 20000) -> None:
    headers = [
        "student_id",
        "age",
        "gender",
        "study_hours_per_day",
        "sleep_hours",
        "phone_usage_hours",
        "social_media_hours",
        "youtube_hours",
        "gaming_hours",
        "breaks_per_day",
        "coffee_intake_mg",
        "exercise_minutes",
        "assignments_completed",
        "attendance_percentage",
        "stress_level",
        "focus_score",
        "final_grade",
        "productivity_score",
    ]

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for student_id in range(1, n_rows + 1):
            age = random.randint(18, 30)
            gender = random.choice(["Male", "Female", "Other"])
            study_hours = round(random.uniform(0.5, 8.0), 2)
            sleep_hours = round(random.uniform(4.0, 9.5), 2)
            phone_usage = round(random.uniform(0.5, 10.0), 2)
            social_media = round(random.uniform(0.0, 6.0), 2)
            youtube = round(random.uniform(0.0, 5.0), 2)
            gaming = round(random.uniform(0.0, 5.0), 2)
            breaks = random.randint(1, 10)
            coffee = random.randint(0, 600)
            exercise = random.randint(0, 120)
            assignments = random.randint(0, 10)
            attendance = round(random.uniform(50.0, 100.0), 2)
            stress = random.randint(1, 10)
            focus = random.randint(1, 100)
            final_grade = round(random.uniform(0.0, 10.0), 2)

            productivity = (
                40
                + study_hours * 6
                + sleep_hours * 3
                + attendance * 0.2
                + exercise * 0.1
                - phone_usage * 2
                - social_media * 1.5
                - gaming * 1.2
                - max(stress - 5, 0) * 2
            )
            productivity = round(max(0.0, min(100.0, productivity)), 2)

            writer.writerow([
                student_id,
                age,
                gender,
                study_hours,
                sleep_hours,
                phone_usage,
                social_media,
                youtube,
                gaming,
                breaks,
                coffee,
                exercise,
                assignments,
                attendance,
                stress,
                focus,
                final_grade,
                productivity,
            ])


def resolver_csv_path(filename: str) -> str:
    # Probamos rutas tipicas para ejecucion local y dentro de Docker.
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        Path.cwd() / filename,
        Path("/opt/spark/work-dir") / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fallback para no bloquear la practica si el dataset no esta disponible.
    synthetic_target = Path(__file__).resolve().parent / filename
    print("[P2][Aviso] Dataset no encontrado. Generando dataset sintetico de ejemplo...")
    generar_dataset_sintetico(synthetic_target)
    print(f"[P2][Aviso] Dataset sintetico creado en: {synthetic_target}")
    return str(synthetic_target)


csv_path = resolver_csv_path(csv_filename)
print(f"[P2] Dataset localizado en: {csv_path}")
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

print("\n[P2] Vamos a comparar cuánto tiempo nos ahorramos definiendo el esquema:")

#Ahora con StructType
t1 = time.time()
df = spark.read.csv(csv_path, header=True, schema=schema)
print(f"  > Con StructType: {time.time() - t1:.4f}s")

# Primero probamos con inferSchema.
t0 = time.time()
df_infer = spark.read.csv(csv_path, header=True, inferSchema=True)
print(f"  > Con inferSchema=True: {time.time() - t0:.4f}s")


# -------------------------------------------------------------
# P3 - Científico de Datos: Transformaciones (Lazy Evaluation)
# -------------------------------------------------------------

print("\n[P3] Aplicando transformaciones")

start_trans = time.time()

# Filtramos por estrés alto. 
df_filtered = df.filter(F.col("stress_level") > 8)

# Metemos una columna nueva para ver quién es productivo de verdad (score > 60).
df_with_new_col = df_filtered.withColumn("es_productivo", F.col("productivity_score") > 60)

# Y nos quedamos solo con las columnas que nos han pedido.
df_final = df_with_new_col.select("student_id", "age", "gender", "stress_level", "es_productivo")

end_trans = time.time()

# -------------------------------------------------------------
# P3.5 - Científico de Datos: Comparación de Tiempos de Ejecución
# -------------------------------------------------------------
print("\n[P3.2] Comparando tiempos reales acumulados de la transformación entre df y df_infer...")

tiempos_df = []
tiempos_df_infer = []
nombres_transformaciones = ["1. Filtro", "2. Columna", "3. Select"]

# 1. Filtro
print("  > Midiendo operación de Filtro...")
t0 = time.time()
df.filter(F.col("stress_level") > 8).count()
tiempos_df.append(time.time() - t0)

t0 = time.time()
df_infer.filter(F.col("stress_level") > 8).count()
tiempos_df_infer.append(time.time() - t0)

# 2. Nueva Columna (sobre el filtro)
print("  > Midiendo operación de Nueva Columna...")
t0 = time.time()
df.filter(F.col("stress_level") > 8).withColumn("es_productivo", F.col("productivity_score") > 60).count()
tiempos_df.append(time.time() - t0)

t0 = time.time()
df_infer.filter(F.col("stress_level") > 8).withColumn("es_productivo", F.col("productivity_score") > 60).count()
tiempos_df_infer.append(time.time() - t0)

# 3. Selección (sobre la nueva columna)
print("  > Midiendo operación de Selección...")
t0 = time.time()
df_final.count()  # df_final ya tiene todo aplicado
tiempos_df.append(time.time() - t0)

t0 = time.time()
df_infer_final = df_infer.filter(F.col("stress_level") > 8)\
                         .withColumn("es_productivo", F.col("productivity_score") > 60)\
                         .select("student_id", "age", "gender", "stress_level", "es_productivo")
df_infer_final.count()
tiempos_df_infer.append(time.time() - t0)

# Generar gráfica de comparación
print(f"\nGenerando gráfica comparativa...")
try:
    import matplotlib.pyplot as plt
    import numpy as np

    grafica_path = str(output_dir / "comparacion_tiempos.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(nombres_transformaciones))
    width = 0.35

    ax.bar(x - width/2, tiempos_df, width, label='StructType (Manual)', color='#4CAF50')
    ax.bar(x + width/2, tiempos_df_infer, width, label='inferSchema (Automático)', color='#F44336')

    ax.set_ylabel('Tiempo de Ejecución Forzada (segundos)')
    ax.set_title('Tiempos Acumulados de Transformaciones: inferSchema vs StructType')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres_transformaciones)
    ax.legend()

    plt.tight_layout()
    plt.savefig(grafica_path)
    print(f"  > ¡Gráfica guardada en {grafica_path}!")
except ImportError:
    print("  > [Aviso] matplotlib o numpy no están instalados. No se ha podido generar la gráfica.")
    print("  > Instálalos con: pip install matplotlib numpy")

# -------------------------------------------------------------
# P4 - Gestor de Calidad: Revisión de Tiempos
# -------------------------------------------------------------
print(f"\n[P4] Resumen de tiempos para el informe:")
print(f"  - Tiempo en 'planificar' las 3 transformaciones: {end_trans - start_trans:.6f}s")

print("\n[P4] Lanzamos ACCIONES.")

# Acción 1: Un count para ver cuántos registros han quedado vivos tras el filtro.
t_act = time.time()
total = df_final.count()
print(f"  1. Acción count(): Tenemos {total} alumnos con estrés. Tardó: {time.time() - t_act:.2f}s")

# Acción 2: Vamos a sacar una muestra por pantalla.
t_act = time.time()
print("  2. Acción show(5):")
df_final.show(5)
print(f"     Tardó: {time.time() - t_act:.2f}s")

# Acción 3: Guardamos el resultado en un CSV nuevo para el profe.
t_act = time.time()
output_spark = str(output_dir / "resultado_estudiantes_spark")
output_fallback = output_dir / "resultado_estudiantes_fallback.csv"
try:
    df_final.write.mode("overwrite").csv(output_spark)
    print(f"  3. Acción write(): Datos guardados en {output_spark}. Tardó: {time.time() - t_act:.2f}s")
except Py4JJavaError as e:
    print("  3. [Aviso] Spark write() ha fallado en Windows (winutils/HADOOP_HOME).")
    print(f"     Detalle: {str(e).splitlines()[0]}")
    with output_fallback.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "age", "gender", "stress_level", "es_productivo"])
        for row in df_final.collect():
            writer.writerow([row["student_id"], row["age"], row["gender"], row["stress_level"], row["es_productivo"]])
    print(f"     Se ha guardado un CSV alternativo en {output_fallback}. Tardó: {time.time() - t_act:.2f}s")

# -------------------------------------------------------------
# P1 - Infraestructura: Chequeo de la Spark UI
# -------------------------------------------------------------
print("\n[P1] Todo listo. He dejado la sesión abierta para que veas el DAG.")
print("  👉 Abre esto en el navegador: http://localhost:4040")
print("  Entra en el último Job y verás el gráfico de cómo Spark ha unido todo.")

# Le meto dos minutos de pausa para que nos dé tiempo a navegar por la UI.
print("\nEsperando 10 segundos antes de matar la sesión (reducido para ejecución rápida)...")
time.sleep(10)

spark.stop()
print("\n--- ¡Todo terminado! A por el siguiente reto. ---")
