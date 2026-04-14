import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    StringType,
)


# ---------------------------
# Configuracion general
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "student_productivity_distraction_dataset_20000.csv")
OUT_DIR = os.path.join(BASE_DIR, "output", "p5_format_benchmark")

PARQUET_PLAIN_PATH = os.path.join(OUT_DIR, "parquet_plain")
PARQUET_SNAPPY_PATH = os.path.join(OUT_DIR, "parquet_snappy")
PARQUET_PART_GENDER_PATH = os.path.join(OUT_DIR, "parquet_partition_gender")
PARQUET_PART_STUDENT_PATH = os.path.join(OUT_DIR, "parquet_partition_student")
PARQUET_MERGE_A_PATH = os.path.join(OUT_DIR, "schema_merge", "part_a")
PARQUET_MERGE_B_PATH = os.path.join(OUT_DIR, "schema_merge", "part_b")
RESULTS_CSV_PATH = os.path.join(OUT_DIR, "metricas_tiempos.csv")


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



def timed(fn):
    start = time.perf_counter()
    value = fn()
    end = time.perf_counter()
    return value, end - start



def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)



def benchmark_source(source_name, load_df_fn):
    df, t_load = timed(load_df_fn)

    def run_query():
        return (
            df.filter((F.col("stress_level") > 8) & (F.col("attendance_percentage") >= 80))
            .agg(
                F.avg("productivity_score").alias("avg_productivity"),
                F.avg("focus_score").alias("avg_focus"),
                F.avg("final_grade").alias("avg_final_grade"),
            )
            .collect()[0]
        )

    result_row, t_query = timed(run_query)

    return {
        "source": source_name,
        "load_s": round(t_load, 6),
        "query_s": round(t_query, 6),
        "total_s": round(t_load + t_query, 6),
        "avg_productivity": float(result_row["avg_productivity"]),
        "avg_focus": float(result_row["avg_focus"]),
        "avg_final_grade": float(result_row["avg_final_grade"]),
    }



def main():
    ensure_dirs()

    spark = (
        SparkSession.builder
        .appName("P5CSVvsParquetSnappyPushdown")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.warehouse.dir", os.path.join(OUT_DIR, "spark-warehouse"))
        .enableHiveSupport()
        .getOrCreate()
    )

    print("=" * 72)
    print("P5 - CSV vs Parquet | Snappy | Predicate Pushdown | Particionado")
    print("=" * 72)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontro el dataset: {CSV_PATH}")

    # 1) Carga base CSV con schema explicito
    print("\n[1/7] Cargando CSV base...")
    df_csv, t_csv_read = timed(lambda: spark.read.csv(CSV_PATH, header=True, schema=schema))
    print(f"  Tiempo lectura CSV inicial: {t_csv_read:.4f}s")

    # 2) Escribir Parquet sin compresion y con Snappy
    print("\n[2/7] Escribiendo Parquet (plain y snappy)...")
    _, t_write_plain = timed(
        lambda: df_csv.write.mode("overwrite").option("compression", "uncompressed").parquet(PARQUET_PLAIN_PATH)
    )
    _, t_write_snappy = timed(
        lambda: df_csv.write.mode("overwrite").option("compression", "snappy").parquet(PARQUET_SNAPPY_PATH)
    )
    print(f"  Escritura Parquet plain:  {t_write_plain:.4f}s")
    print(f"  Escritura Parquet snappy: {t_write_snappy:.4f}s")

    # 3) Benchmark CSV vs Parquet
    print("\n[3/7] Comparando tiempos de carga + filtro + medias...")
    results = []

    results.append(
        benchmark_source(
            "csv",
            lambda: spark.read.csv(CSV_PATH, header=True, schema=schema),
        )
    )
    results.append(
        benchmark_source(
            "parquet_plain",
            lambda: spark.read.parquet(PARQUET_PLAIN_PATH),
        )
    )
    results.append(
        benchmark_source(
            "parquet_snappy",
            lambda: spark.read.parquet(PARQUET_SNAPPY_PATH),
        )
    )

    for row in results:
        print(
            f"  {row['source']:<15} load={row['load_s']:.4f}s "
            f"query={row['query_s']:.4f}s total={row['total_s']:.4f}s"
        )

    # 4) Predicate pushdown en Parquet
    print("\n[4/7] Predicate pushdown (revisar plan fisico):")
    df_parquet = spark.read.parquet(PARQUET_SNAPPY_PATH)
    filtered = df_parquet.filter((F.col("stress_level") > 8) & (F.col("attendance_percentage") >= 80))
    filtered.explain(True)

    # 5) Particionado y cardinalidad
    print("\n[5/7] Particionado por baja vs alta cardinalidad...")
    _, t_part_gender = timed(
        lambda: df_csv.write.mode("overwrite").partitionBy("gender").parquet(PARQUET_PART_GENDER_PATH)
    )
    _, t_part_student = timed(
        lambda: df_csv.write.mode("overwrite").partitionBy("student_id").parquet(PARQUET_PART_STUDENT_PATH)
    )
    print(f"  partitionBy(gender)     -> {t_part_gender:.4f}s")
    print(f"  partitionBy(student_id) -> {t_part_student:.4f}s")

    # 6) Schema merging
    print("\n[6/7] Schema merging demo...")
    df_a = df_csv.filter(F.col("student_id") <= 10000)
    df_b = df_csv.filter(F.col("student_id") > 10000).withColumn(
        "score_compuesto", F.col("focus_score") + F.col("stress_level")
    )

    df_a.write.mode("overwrite").parquet(PARQUET_MERGE_A_PATH)
    df_b.write.mode("overwrite").parquet(PARQUET_MERGE_B_PATH)

    merged = spark.read.option("mergeSchema", "true").parquet(PARQUET_MERGE_A_PATH, PARQUET_MERGE_B_PATH)
    print("  Esquema tras mergeSchema=true:")
    merged.printSchema()

    # 7) Bucketing (puede depender del entorno)
    print("\n[7/7] Bucketing (opcional, enfocado a joins)...")
    try:
        spark.sql("DROP TABLE IF EXISTS p5_bucketed_students")
        _, t_bucket = timed(
            lambda: (
                df_csv.write.mode("overwrite")
                .bucketBy(8, "student_id")
                .sortBy("student_id")
                .saveAsTable("p5_bucketed_students")
            )
        )
        print(f"  Tabla bucketed creada en {t_bucket:.4f}s")

        # Join de ejemplo para notar comportamiento
        left = spark.table("p5_bucketed_students").select("student_id", "focus_score")
        right = spark.table("p5_bucketed_students").select("student_id", "productivity_score")

        _, t_join = timed(lambda: left.join(right, on="student_id", how="inner").count())
        print(f"  Join sobre tabla bucketed -> {t_join:.4f}s")
    except Exception as ex:
        print(f"  Aviso: bucketing no disponible o fallo de entorno: {str(ex).splitlines()[0]}")

    # Persistir metricas principales
    metrics_df = spark.createDataFrame(results)
    metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(RESULTS_CSV_PATH)

    print("\nResumen de metricas guardado en:")
    print(f"  {RESULTS_CSV_PATH}")

    print("\nProceso completado.")
    spark.stop()


if __name__ == "__main__":
    main()
