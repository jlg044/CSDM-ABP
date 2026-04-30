
import time
import os
import glob
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, FloatType
)
from pyspark.sql import functions as F


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — umbrales ajustados sobre percentiles reales del dataset
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVO_CSV    = "student_productivity_distraction_CLEAN.csv"
ARCHIVO_SALIDA = "student_productivity_distraction_GAMER_v2.csv"

# Puntuación base (gaming_hours)
BASE_NOVATO   = 1.0
BASE_GAMER    = 2.0
BASE_EXPERTO  = 3.0
LIM_NOVATO    = 2.0   # ≤ 2h  → base Novato
LIM_GAMER     = 4.0   # ≤ 4h  → base Gamer

# Umbrales para bonificaciones (basados en percentiles del dataset)
UMBRAL_YOUTUBE       = 3.5   # p50 = 2.98 → "por encima de la media"
UMBRAL_SOCIAL_MEDIA  = 5.0   # p50 = 4.01 → "por encima de la media"
UMBRAL_SLEEP_BAJO    = 5.0   # p25 = 4.78 → "sueño escaso"
UMBRAL_PHONE         = 8.0   # p75 = 9.1  → "uso intensivo del móvil"
UMBRAL_STRESS        = 8     # p75 = 8.0  → "estrés alto"
EDAD_JOVEN           = 20

# Bonificaciones por columna
BONO_YOUTUBE      = 0.5
BONO_SOCIAL       = 0.3
BONO_SLEEP        = 0.5
BONO_PHONE        = 0.3
BONO_STRESS       = 0.2
BONO_EDAD_JOVEN   = 0.3
BONO_GENDER_OTHER = 0.1

# Umbrales de clasificación final
SCORE_NOVATO   = 0.0   # > 0
SCORE_GAMER    = 1.5
SCORE_EXPERTO  = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# ESQUEMA
# ─────────────────────────────────────────────────────────────────────────────

schema = StructType([
    StructField("student_id",             IntegerType(), True),
    StructField("age",                    IntegerType(), True),
    StructField("gender",                 StringType(),  True),
    StructField("study_hours_per_day",    FloatType(),   True),
    StructField("sleep_hours",            FloatType(),   True),
    StructField("phone_usage_hours",      FloatType(),   True),
    StructField("social_media_hours",     FloatType(),   True),
    StructField("youtube_hours",          FloatType(),   True),
    StructField("gaming_hours",           FloatType(),   True),
    StructField("breaks_per_day",         IntegerType(), True),
    StructField("coffee_intake_mg",       IntegerType(), True),
    StructField("exercise_minutes",       IntegerType(), True),
    StructField("assignments_completed",  IntegerType(), True),
    StructField("attendance_percentage",  FloatType(),   True),
    StructField("stress_level",           IntegerType(), True),
    StructField("focus_score",            IntegerType(), True),
    StructField("final_grade",            FloatType(),   True),
    StructField("productivity_score",     FloatType(),   True),
])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CLASIFICACIÓN PERFIL GAMER v2 — 8 columnas — PySpark withColumn")
    print("=" * 70)

    # ── Iniciar Spark ─────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("PerfilGamerV2")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print(f"\n  Cargando: {ARCHIVO_CSV}")
    df = spark.read.csv(ARCHIVO_CSV, header=True, schema=schema)
    n_filas = df.count()
    print(f"  Filas cargadas: {n_filas:,}")

    # ── Descripción de la lógica ──────────────────────────────────────────────
    print("\n  Columnas y bonificaciones:")
    print("  ┌─────────────────────────────────┬──────────┬──────────────────────────┐")
    print("  │ Columna                         │   Bono   │ Condición                │")
    print("  ├─────────────────────────────────┼──────────┼──────────────────────────┤")
    print(f"  │ gaming_hours (0h)               │  0.0 pts │ → No Juega               │")
    print(f"  │ gaming_hours (0-2h)             │  1.0 pts │ base Novato              │")
    print(f"  │ gaming_hours (2-4h)             │  2.0 pts │ base Gamer               │")
    print(f"  │ gaming_hours (>4h)              │  3.0 pts │ base Experto             │")
    print(f"  │ youtube_hours                   │ +{BONO_YOUTUBE} pts │ > {UMBRAL_YOUTUBE}h/día               │")
    print(f"  │ social_media_hours              │ +{BONO_SOCIAL} pts │ > {UMBRAL_SOCIAL_MEDIA}h/día               │")
    print(f"  │ sleep_hours                     │ +{BONO_SLEEP} pts │ < {UMBRAL_SLEEP_BAJO}h/día (poco sueño)  │")
    print(f"  │ phone_usage_hours               │ +{BONO_PHONE} pts │ > {UMBRAL_PHONE}h/día               │")
    print(f"  │ stress_level                    │ +{BONO_STRESS} pts │ >= {UMBRAL_STRESS}/10                 │")
    print(f"  │ age                             │ +{BONO_EDAD_JOVEN} pts │ <= {EDAD_JOVEN} años               │")
    print(f"  │ gender                          │ +{BONO_GENDER_OTHER} pts │ == 'Other'               │")
    print(f"  └─────────────────────────────────┴──────────┴──────────────────────────┘")
    print(f"  Clasificación: 0→No Juega | <1.5→Novato | <2.5→Gamer | ≥2.5→Experto")

    # ── WITHCOLUMN 1: score_gamer ─────────────────────────────────────────────
    t_inicio = time.perf_counter()

    df_scored = df.withColumn(
        "score_gamer",
        F.when(
            F.col("gaming_hours") == 0,
            F.lit(0.0)                               # No Juega: sin puntos
        ).otherwise(
            # Puntuación base según horas de juego
            F.when(F.col("gaming_hours") <= LIM_NOVATO, F.lit(BASE_NOVATO))
             .when(F.col("gaming_hours") <= LIM_GAMER,  F.lit(BASE_GAMER))
             .otherwise(F.lit(BASE_EXPERTO))

            # + Bono YouTube: consumo de contenido gaming
            + F.when(F.col("youtube_hours") > UMBRAL_YOUTUBE,
                     F.lit(BONO_YOUTUBE)).otherwise(F.lit(0.0))

            # + Bono redes sociales: ecosistema digital gamer
            + F.when(F.col("social_media_hours") > UMBRAL_SOCIAL_MEDIA,
                     F.lit(BONO_SOCIAL)).otherwise(F.lit(0.0))

            # + Bono sueño escaso: sacrifica descanso por gaming
            + F.when(F.col("sleep_hours") < UMBRAL_SLEEP_BAJO,
                     F.lit(BONO_SLEEP)).otherwise(F.lit(0.0))

            # + Bono uso del móvil: alto engagement digital
            + F.when(F.col("phone_usage_hours") > UMBRAL_PHONE,
                     F.lit(BONO_PHONE)).otherwise(F.lit(0.0))

            # + Bono estrés alto: usa gaming como válvula de escape
            + F.when(F.col("stress_level") >= UMBRAL_STRESS,
                     F.lit(BONO_STRESS)).otherwise(F.lit(0.0))

            # + Bono edad joven: mayor dedicación con menos horas
            + F.when(F.col("age") <= EDAD_JOVEN,
                     F.lit(BONO_EDAD_JOVEN)).otherwise(F.lit(0.0))

            # + Bono género Other
            + F.when(F.col("gender") == "Other",
                     F.lit(BONO_GENDER_OTHER)).otherwise(F.lit(0.0))
        )
    )

    # ── WITHCOLUMN 2: perfil_gamer ────────────────────────────────────────────
    df_clasificado = df_scored.withColumn(
        "perfil_gamer",
        F.when(F.col("score_gamer") == 0,                    F.lit("No Juega"))
         .when(F.col("score_gamer") >= SCORE_EXPERTO,        F.lit("Experto Gamer"))
         .when(F.col("score_gamer") >= SCORE_GAMER,          F.lit("Gamer"))
         .otherwise(                                          F.lit("Novato"))
    )

    # Materializar (Spark es lazy — count() fuerza la evaluación del DAG)
    n_resultado = df_clasificado.count()

    t_fin    = time.perf_counter()
    t_ms     = (t_fin - t_inicio) * 1_000
    t_por_f  = t_ms / n_resultado if n_resultado else 0

    # ── Resultados ────────────────────────────────────────────────────────────
    print("\n  Distribución de perfil_gamer:")
    print("  " + "─" * 55)
    df_clasificado.groupBy("perfil_gamer") \
        .agg(
            F.count("*").alias("total"),
            F.round(F.avg("score_gamer"), 3).alias("score_medio"),
            F.round(F.avg("gaming_hours"), 2).alias("gaming_h_medio"),
            F.round(F.avg("age"), 1).alias("edad_media")
        ) \
        .orderBy(F.col("score_medio").desc()) \
        .show(truncate=False)

    print("  Distribución cruzada: género × perfil_gamer")
    print("  " + "─" * 55)
    df_clasificado \
        .groupBy("gender") \
        .pivot("perfil_gamer", ["No Juega", "Novato", "Gamer", "Experto Gamer"]) \
        .count() \
        .orderBy("gender") \
        .show(truncate=False)

    print("  Score medio por bono activado:")
    print("  " + "─" * 55)
    df_clasificado \
        .groupBy("perfil_gamer") \
        .agg(
            F.round(F.avg("youtube_hours"), 2).alias("yt_h_med"),
            F.round(F.avg("social_media_hours"), 2).alias("sm_h_med"),
            F.round(F.avg("sleep_hours"), 2).alias("sleep_med"),
            F.round(F.avg("phone_usage_hours"), 2).alias("phone_med"),
            F.round(F.avg("stress_level"), 2).alias("stress_med")
        ) \
        .orderBy("perfil_gamer") \
        .show(truncate=False)

    # ── Tiempo ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  ✓ Filas procesadas         : {n_resultado:,}")
    print(f"  ✓ withColumn x2 + eval     : {t_ms:.2f} ms  ({t_ms/1000:.3f} s)")
    print(f"  ✓ Tiempo por fila          : {t_por_f:.4f} ms/fila")
    print("=" * 70)

    # ── Guardar ───────────────────────────────────────────────────────────────
    print(f"\n  Guardando: {ARCHIVO_SALIDA}")
    tmp = ARCHIVO_SALIDA + "_tmp"
    df_clasificado.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(tmp)

    partes = glob.glob(f"{tmp}/part-*.csv")
    if partes:
        shutil.copy(partes[0], ARCHIVO_SALIDA)
        shutil.rmtree(tmp)
        print("Guardado correctamente")
    else:
        print("No se encontró la parte generada por Spark")

    spark.stop()


if __name__ == "__main__":
    main()
