import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import functions as F

# Configuro Spark
spark = SparkSession.builder \
    .appName("SparkRolesABP") \
    .master("local[*]") \
    .getOrCreate()

print("="*50)
print("INICIO DEL PROCESO")
print("="*50)

# Carga de datos con esquema manual
print("\nCargando el fichero y el esquema...")

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

# Path relativo al archivo CSV
csv_path = "student_productivity_distraction_dataset_20000.csv"
df = spark.read.csv(csv_path, header=True, schema=schema)

print("Esquema cargado correctamente.")
df.printSchema()

# Aplicamos transformaciones (Lazy)
print("\nFiltramos y seleccionamos columnas...")

# Transformación 1: Filtrar por nivel de estrés alto
# Transformación 2: Añadir columna calculada (withColumn)
# Transformación 3: Seleccionar columnas clave
start_trans = time.time()
df_with_col = df.filter(F.col("stress_level") > 8) \
                .withColumn("es_productivo", F.col("productivity_score") > 60)

df_transformed = df_with_col.select("student_id", "age", "gender", "stress_level", "es_productivo")
end_trans = time.time()

# Medimos tiempos para el informe
print("\nChequeando tiempos...")
print(f"Planificación de transformaciones: {end_trans - start_trans:.5f}s")
print("Como es Lazy Evaluation, aquí Spark aún no ha hecho nada 'real'.")

print("\nEjecutando la primera ACCIÓN (count)...")
start_action1 = time.time()
total_high_stress = df_transformed.count()
end_action1 = time.time()
print(f"Resultados de count(): {total_high_stress} estudiantes con estrés > 8.")
print(f"Tiempo de la acción count(): {end_action1 - start_action1:.2f} segundos.")

print("\nEjecutando la segunda ACCIÓN (show)...")
start_action2 = time.time()
df_transformed.show(5)
end_action2 = time.time()
print(f"Tiempo de la acción show(): {end_action2 - start_action2:.2f} segundos.")

print("\nEjecutando la tercera ACCIÓN (collect)...")
start_action3 = time.time()
rows = df_transformed.collect()
end_action3 = time.time()
print(f"Resultados de collect(): Obtenidas {len(rows)} filas.")
print(f"Tiempo de la acción collect(): {end_action3 - start_action3:.2f} segundos.")

print("\nEjecutando la cuarta ACCIÓN (write)...")
start_action4 = time.time()
output_path = "output/resultado_message_spark"
try:
    df_transformed.write.mode("overwrite").csv(output_path)
    print(f"Datos guardados correctamente en {output_path}.")
except Exception as e:
    print(f"Falla write() por entorno: {str(e).splitlines()[0]}")
end_action4 = time.time()
print(f"Tiempo de la acción write(): {end_action4 - start_action4:.2f} segundos.")

# Info para ver la Spark UI
print("\nChequeo de la UI en: http://localhost:4040")
print("-" * 50)
print("La aplicación sigue activa. Por favor, abre en tu navegador:")
print(">>> http://localhost:4040 <<<")
print("-" * 50)
print("Instrucciones para P1:")
print("1. Ve a la pestaña 'Jobs'. Verás que hay 4 Jobs (uno por cada Acción: count, show, collect y write).")
print("2. Haz clic en un Job para ver el DAG (Directed Acyclic Graph).")
print("3. Observa cómo las etapas (Stages) se dividen según las transformaciones aplicadas.")
print("-" * 50)

print("\nEsperando 120 segundos para que puedas revisar la Spark UI en http://localhost:4040...")
time.sleep(120)

spark.stop()
print("\nSesión de Spark cerrada. ¡Reto completado!")
