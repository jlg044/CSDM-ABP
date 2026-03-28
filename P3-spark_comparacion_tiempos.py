import time
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar gráficas sin ventana
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import functions as F

# ============================================================
# Configuración de SparkSession
# ============================================================
spark = SparkSession.builder \
    .appName("ComparacionInferSchemaVsStructType") \
    .master("local[*]") \
    .getOrCreate()

csv_path = "student_productivity_distraction_dataset_20000.csv"

print("=" * 60)
print("  COMPARACIÓN DE RENDIMIENTO: inferSchema vs StructType")
print("=" * 60)


# ============================================================
# Funciones auxiliares para transformaciones y acciones
# ============================================================

def ejecutar_transformaciones(df):
    """Ejecuta las transformaciones solicitadas (filter, select, withColumn)."""
    tiempos = {}

    # Transformación 1: Filter – estudiantes con estrés > 8
    start = time.time()
    df_filter = df.filter(F.col("stress_level") > 8)
    tiempos["filter"] = time.time() - start

    # Transformación 2: Select – columnas clave
    start = time.time()
    df_select = df.select("student_id", "age", "gender", "stress_level",
                          "focus_score", "productivity_score")
    tiempos["select"] = time.time() - start

    # Transformación 3: withColumn – nueva columna calculada
    start = time.time()
    df_with = df.withColumn("ratio_estudio_sueño",
                            F.col("study_hours_per_day") / F.col("sleep_hours"))
    tiempos["withColumn"] = time.time() - start

    return tiempos, {
        "filter": df_filter,
        "select": df_select,
        "withColumn": df_with,
    }


def ejecutar_acciones(dfs, df_original):
    """Ejecuta las acciones solicitadas (count, show, collect, write)."""
    tiempos = {}

    # Acción 1: count() sobre el df filtrado
    start = time.time()
    count_result = dfs["filter"].count()
    tiempos["count (filter)"] = time.time() - start
    print(f"    count (filter): {count_result} filas")

    # Acción 2: show() sobre select
    start = time.time()
    dfs["select"].show(5)
    tiempos["show (select)"] = time.time() - start

    # Acción 3: collect() sobre withColumn
    start = time.time()
    collect_result = dfs["withColumn"].limit(5).collect()
    tiempos["collect (withColumn)"] = time.time() - start
    print(f"    collect: {len(collect_result)} filas recuperadas")

    # Acción 4: write sobre select (parquet)
    start = time.time()
    dfs["select"].write.mode("overwrite").parquet("resultado_comparacion.parquet")
    tiempos["write (parquet)"] = time.time() - start
    print("    write: completado")

    return tiempos


# ============================================================
# 1. CARGA CON inferSchema
# ============================================================
print("\n" + "=" * 60)
print("  FASE 1: Carga con inferSchema=True")
print("=" * 60)

start_carga = time.time()
df_infer = spark.read.csv(csv_path, header=True, inferSchema=True)
tiempo_carga_infer = time.time() - start_carga
print(f"\n  Tiempo de carga (inferSchema): {tiempo_carga_infer:.4f} s")
df_infer.printSchema()

# --- Transformaciones con inferSchema ---
print("\n  >> Transformaciones (inferSchema)...")
tiempos_trans_infer, dfs_infer = ejecutar_transformaciones(df_infer)
for nombre, t in tiempos_trans_infer.items():
    print(f"    {nombre}: {t:.6f} s")

# --- Acciones con inferSchema ---
print("\n  >> Acciones (inferSchema)...")
tiempos_acc_infer = ejecutar_acciones(dfs_infer, df_infer)
for nombre, t in tiempos_acc_infer.items():
    print(f"    {nombre}: {t:.4f} s")


# ============================================================
# 2. CARGA CON StructType
# ============================================================
print("\n" + "=" * 60)
print("  FASE 2: Carga con StructType (esquema explícito)")
print("=" * 60)

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

start_carga = time.time()
df_struct = spark.read.csv(csv_path, header=True, schema=schema)
tiempo_carga_struct = time.time() - start_carga
print(f"\n  Tiempo de carga (StructType): {tiempo_carga_struct:.4f} s")
df_struct.printSchema()

# --- Transformaciones con StructType ---
print("\n  >> Transformaciones (StructType)...")
tiempos_trans_struct, dfs_struct = ejecutar_transformaciones(df_struct)
for nombre, t in tiempos_trans_struct.items():
    print(f"    {nombre}: {t:.6f} s")

# --- Acciones con StructType ---
print("\n  >> Acciones (StructType)...")
tiempos_acc_struct = ejecutar_acciones(dfs_struct, df_struct)
for nombre, t in tiempos_acc_struct.items():
    print(f"    {nombre}: {t:.4f} s")


# ============================================================
# 3. GENERACIÓN DE GRÁFICAS COMPARATIVAS
# ============================================================
print("\n" + "=" * 60)
print("  GENERANDO GRÁFICAS COMPARATIVAS...")
print("=" * 60)

# --- Estilo general ---
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eaeaea',
    'text.color': '#eaeaea',
    'xtick.color': '#eaeaea',
    'ytick.color': '#eaeaea',
    'font.size': 11,
})

# --------------------------------------------------------
# GRÁFICA 1: Comparación de Transformaciones
# --------------------------------------------------------
nombres_trans = list(tiempos_trans_infer.keys())
vals_infer_t = [tiempos_trans_infer[n] for n in nombres_trans]
vals_struct_t = [tiempos_trans_struct[n] for n in nombres_trans]

x = np.arange(len(nombres_trans))
ancho = 0.35

fig1, ax1 = plt.subplots(figsize=(12, 6))
bars1 = ax1.bar(x - ancho / 2, vals_infer_t, ancho, label='inferSchema',
                color='#e94560', edgecolor='#fff', linewidth=0.5, zorder=3)
bars2 = ax1.bar(x + ancho / 2, vals_struct_t, ancho, label='StructType',
                color='#0f3460', edgecolor='#fff', linewidth=0.5, zorder=3)

ax1.set_xlabel('Transformación')
ax1.set_ylabel('Tiempo (segundos)')
ax1.set_title('Comparación de Tiempos – Transformaciones\ninferSchema vs StructType',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(nombres_trans, rotation=20, ha='right')
ax1.legend(loc='upper right', framealpha=0.8)
ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

# Añadir valores encima de las barras
for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f'{h:.5f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8, color='#e94560')
for bar in bars2:
    h = bar.get_height()
    ax1.annotate(f'{h:.5f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8, color='#53a8ff')

plt.tight_layout()
fig1.savefig("grafica_transformaciones.png", dpi=150, bbox_inches='tight')
print("  ✅ Guardada: grafica_transformaciones.png")

# --------------------------------------------------------
# GRÁFICA 2: Comparación de Acciones
# --------------------------------------------------------
nombres_acc = list(tiempos_acc_infer.keys())
vals_infer_a = [tiempos_acc_infer[n] for n in nombres_acc]
vals_struct_a = [tiempos_acc_struct[n] for n in nombres_acc]

x2 = np.arange(len(nombres_acc))

fig2, ax2 = plt.subplots(figsize=(12, 6))
bars3 = ax2.bar(x2 - ancho / 2, vals_infer_a, ancho, label='inferSchema',
                color='#e94560', edgecolor='#fff', linewidth=0.5, zorder=3)
bars4 = ax2.bar(x2 + ancho / 2, vals_struct_a, ancho, label='StructType',
                color='#0f3460', edgecolor='#fff', linewidth=0.5, zorder=3)

ax2.set_xlabel('Acción')
ax2.set_ylabel('Tiempo (segundos)')
ax2.set_title('Comparación de Tiempos – Acciones\ninferSchema vs StructType',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(nombres_acc, rotation=20, ha='right')
ax2.legend(loc='upper right', framealpha=0.8)
ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

for bar in bars3:
    h = bar.get_height()
    ax2.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8, color='#e94560')
for bar in bars4:
    h = bar.get_height()
    ax2.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=8, color='#53a8ff')

plt.tight_layout()
fig2.savefig("grafica_acciones.png", dpi=150, bbox_inches='tight')
print("  ✅ Guardada: grafica_acciones.png")


# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 60)
print("  RESUMEN DE TIEMPOS")
print("=" * 60)
print(f"\n  {'Operación':<25} {'inferSchema':>12} {'StructType':>12} {'Diferencia':>12}")
print("  " + "-" * 61)

print("\n  --- Carga ---")
print(f"  {'Carga CSV':<25} {tiempo_carga_infer:>12.4f} {tiempo_carga_struct:>12.4f} {tiempo_carga_infer - tiempo_carga_struct:>+12.4f}")

print("\n  --- Transformaciones ---")
for nombre in nombres_trans:
    ti = tiempos_trans_infer[nombre]
    ts = tiempos_trans_struct[nombre]
    print(f"  {nombre:<25} {ti:>12.6f} {ts:>12.6f} {ti - ts:>+12.6f}")

print("\n  --- Acciones ---")
for nombre in nombres_acc:
    ti = tiempos_acc_infer[nombre]
    ts = tiempos_acc_struct[nombre]
    print(f"  {nombre:<25} {ti:>12.4f} {ts:>12.4f} {ti - ts:>+12.4f}")

print("\n" + "=" * 60)
print("  Gráficas guardadas en el directorio actual.")
print("=" * 60)

spark.stop()
print("\n  Sesión de Spark cerrada. ¡Comparación completada!")
