"""
=============================================================================
  FEATURE SELECTION CON CORRELACIÓN
  Dataset: student_productivity_distraction_CLEAN.csv
  Autor: Juan (generado con Cowork / Claude)
  Fecha: 2026-04-21
=============================================================================

  OBJETIVO:
  ─────────────────────────────────────────────────────────────────────────
  • Cargar datos limpios del pipeline anterior (P4)
  • Transformar features categóricas: OneHotEncoder (gender)
  • Calcular CORRELACIÓN de cada feature con variable objetivo (productivity_score)
  • Seleccionar automáticamente TOP N features por correlación
  • Normalizar con StandardScaler (Scikit-learn)
  • Visualizar correlaciones
  • Generar reporte de features seleccionadas
  ─────────────────────────────────────────────────────────────────────────

  FEATURES DEL DATASET:
  ─────────────────────────────────────────────────────────────────────────
  Numéricas:
    • age, study_hours_per_day, sleep_hours, phone_usage_hours
    • social_media_hours, youtube_hours, gaming_hours, breaks_per_day
    • coffee_intake_mg, exercise_minutes, assignments_completed
    • attendance_percentage, stress_level, focus_score, final_grade
    
  Categórica:
    • gender → OneHotEncoder
    
  Variable Objetivo:
    • productivity_score (0-100)
  ─────────────────────────────────────────────────────────────────────────

  MODO DE USO:
      python P6-feature_selection_correlation.py

=============================================================================
"""

import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "student_productivity_distraction_CLEAN.csv")
OUT_DIR = os.path.join(BASE_DIR, "output", "p6_feature_selection")

# Crear directorio de salida
os.makedirs(OUT_DIR, exist_ok=True)

CORRELATIONS_CSV = os.path.join(OUT_DIR, "correlaciones.csv")
TRANSFORMED_PARQUET = os.path.join(OUT_DIR, "datos_transformados")
FEATURE_REPORT = os.path.join(OUT_DIR, "reporte_features_seleccionadas.txt")
CORRELATION_PLOT = os.path.join(OUT_DIR, "correlacion_features.png")
SELECTED_FEATURES_PLOT = os.path.join(OUT_DIR, "top_features.png")

# Número de features a seleccionar (TOP N)
TOP_N_FEATURES = 10

# Schema del dataset limpio
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

# Features numéricas (excluyen student_id que es solo identificador)
NUMERIC_FEATURES = [
    "age", "study_hours_per_day", "sleep_hours", "phone_usage_hours",
    "social_media_hours", "youtube_hours", "gaming_hours", "breaks_per_day",
    "coffee_intake_mg", "exercise_minutes", "assignments_completed",
    "attendance_percentage", "stress_level", "focus_score", "final_grade"
]

# Feature categórica
CATEGORICAL_FEATURES = ["gender"]

TARGET = "productivity_score"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def calculate_correlations_spark(spark, df, numeric_features):
    """
    Calcula correlación usando SQL para manejar NULLs
    """
    print("\n[*] Calculando correlaciones con Spark SQL...")
    
    corr_data = []
    
    # Para cada feature, calcular correlación con target ignorando nulos
    for feature in numeric_features:
        query = f"""
        SELECT 
            corr({feature}, {TARGET}) as correlation
        FROM df_temp
        WHERE {feature} IS NOT NULL AND {TARGET} IS NOT NULL
        """
        
        df.createOrReplaceTempView("df_temp")
        result = spark.sql(query).collect()
        
        if result and result[0][0] is not None:
            corr_value = float(result[0][0])
        else:
            corr_value = 0.0
        
        corr_data.append({
            'feature': feature,
            'correlation': corr_value,
            'abs_correlation': abs(corr_value)
        })
    
    # Convertir a Spark DataFrame y ordenar
    corr_spark_df = spark.createDataFrame(corr_data)
    corr_sorted = corr_spark_df.orderBy(F.desc("abs_correlation"))
    
    return corr_sorted


def plot_correlations(corr_list):
    """Genera gráfica de correlaciones desde lista de diccionarios"""
    features = [item['feature'] for item in corr_list]
    correlations = [item['correlation'] for item in corr_list]
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red' if x < 0 else 'green' for x in correlations]
    
    plt.barh(features, correlations, color=colors, alpha=0.7)
    plt.xlabel('Correlación con Productivity Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Correlación de Features con Variable Objetivo\n(Rojo=Negativa, Verde=Positiva)', 
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CORRELATION_PLOT, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de correlaciones guardada en: {CORRELATION_PLOT}")
    plt.close()


def plot_top_features(top_corr_list):
    """Genera gráfica de TOP N features"""
    features = [item['feature'] for item in top_corr_list]
    correlations = [item['correlation'] for item in top_corr_list]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['red' if x < 0 else 'green' for x in correlations]
    
    plt.barh(features, correlations, color=colors, alpha=0.8)
    plt.xlabel('Correlación con Productivity Score', fontsize=11, fontweight='bold')
    plt.ylabel('Feature', fontsize=11, fontweight='bold')
    plt.title(f'TOP {TOP_N_FEATURES} Features Seleccionados por Correlación', 
              fontsize=13, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SELECTED_FEATURES_PLOT, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de TOP features guardada en: {SELECTED_FEATURES_PLOT}")
    plt.close()


def generate_report(all_corr_list, top_corr_list):
    """Genera reporte textual de feature selection"""
    
    # Limpiar archivo anterior
    if os.path.exists(FEATURE_REPORT):
        os.remove(FEATURE_REPORT)
    
    with open(FEATURE_REPORT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE FEATURE SELECTION - CORRELACIÓN\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Variable Objetivo: {TARGET}\n")
        f.write(f"Total de features analizadas: {len(all_corr_list)}\n")
        f.write(f"Features seleccionados (TOP N): {TOP_N_FEATURES}\n\n")
        
        # TOP features
        f.write("-"*80 + "\n")
        f.write("TOP FEATURES SELECCIONADOS (por valor absoluto de correlación)\n")
        f.write("-"*80 + "\n\n")
        
        for idx, item in enumerate(top_corr_list, 1):
            f.write(f"{idx}. {item['feature']:<30} | Correlación: {item['correlation']:>8.4f} | "
                   f"Abs: {item['abs_correlation']:>7.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("INTERPRETACIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write("• Correlación > 0: Feature aumenta con productivity_score\n")
        f.write("• Correlación < 0: Feature disminuye con productivity_score\n")
        f.write("• |Correlación| cercano a 1: Relación muy fuerte\n")
        f.write("• |Correlación| cercano a 0: Relación muy débil\n\n")
        
        # Estadísticas
        correlations = [item['correlation'] for item in all_corr_list]
        abs_correlations = [item['abs_correlation'] for item in all_corr_list]
        
        f.write("ESTADÍSTICAS DE CORRELACIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Correlación máxima (positiva):  {max(correlations):>8.4f}\n")
        f.write(f"Correlación mínima (negativa):  {min(correlations):>8.4f}\n")
        f.write(f"Correlación media (absoluta):   {sum(abs_correlations)/len(abs_correlations):>8.4f}\n\n")
        
        # Features positivos vs negativos
        f.write("CLASIFICACIÓN\n")
        f.write("-"*80 + "\n\n")
        
        positive = [item for item in all_corr_list if item['correlation'] > 0]
        negative = [item for item in all_corr_list if item['correlation'] < 0]
        
        f.write(f"Features CON correlación positiva (impactan positivamente): {len(positive)}\n")
        for item in sorted(positive, key=lambda x: x['correlation'], reverse=True)[:5]:
            f.write(f"  • {item['feature']:<30} {item['correlation']:>8.4f}\n")
        
        f.write(f"\nFeatures CON correlación negativa (impactan negativamente): {len(negative)}\n")
        for item in sorted(negative, key=lambda x: x['correlation'])[:5]:
            f.write(f"  • {item['feature']:<30} {item['correlation']:>8.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("  FEATURE SELECTION CON CORRELACIÓN - PIPELINE P6")
    print("="*80 + "\n")
    
    # Inicializar Spark
    print("[1] Inicializando Spark...")
    spark = SparkSession.builder \
        .appName("Feature_Selection_Correlation") \
        .getOrCreate()
    
    try:
        # Cargar datos limpios
        print(f"\n[2] Cargando dataset limpio desde: {CLEAN_CSV_PATH}")
        if not os.path.exists(CLEAN_CSV_PATH):
            print(f"❌ ERROR: No se encontró el archivo: {CLEAN_CSV_PATH}")
            print("   Asegúrate de haber ejecutado P4-pipeline_limpieza.py primero")
            return
        
        df = spark.read.schema(schema).csv(CLEAN_CSV_PATH, header=True)
        df.cache()
        
        count = df.count()
        print(f"✓ Dataset cargado: {count} registros")
        
        # Mostrar primeras filas
        print("\nPrimeras 3 filas del dataset:")
        df.show(3, truncate=False)
        
        # TRANSFORMACIÓN INICIAL: Calcular correlaciones con SQL
        print("\n[3] Preparando para calcular correlaciones...")
        
        # Calcular correlaciones con Spark SQL
        print("\n[4] Calculando correlaciones con Spark SQL...")
        corr_sorted = calculate_correlations_spark(spark, df, NUMERIC_FEATURES)
        
        # Convertir a lista local para análisis
        corr_list = corr_sorted.collect()
        corr_list = [row.asDict() for row in corr_list]
        
        # Guardar correlaciones a CSV
        corr_sorted.coalesce(1).write.csv(CORRELATIONS_CSV, header=True, mode="overwrite")
        print(f"✓ Correlaciones guardadas en: {CORRELATIONS_CSV}")
        
        print("\nTodas las correlaciones:")
        print("─"*80)
        for item in corr_list:
            print(f"{item['feature']:<30} Correlación: {item['correlation']:>8.4f} | Abs: {item['abs_correlation']:>7.4f}")
        
        # Seleccionar TOP N
        print(f"\n[5] Seleccionando TOP {TOP_N_FEATURES} features...")
        top_corr_list = corr_list[:TOP_N_FEATURES]
        
        print(f"\n✓ TOP {TOP_N_FEATURES} Features Seleccionados:")
        for idx, item in enumerate(top_corr_list, 1):
            print(f"  {idx}. {item['feature']:<30} Correlación: {item['correlation']:>8.4f}")
        
        # Generar visualizaciones
        print("\n[6] Generando visualizaciones...")
        plot_correlations(corr_list)
        plot_top_features(top_corr_list)
        
        # Generar reporte
        print("\n[7] Generando reporte...")
        generate_report(corr_list, top_corr_list)
        print(f"✓ Reporte guardado en: {FEATURE_REPORT}")
        
        # TRANSFORMACIÓN FINAL: StringIndexer + OneHotEncoder + StandardScaler con features seleccionados
        print("\n[8] Transformando y normalizando features seleccionados...")
        
        selected_numeric_features = [item['feature'] for item in top_corr_list]
        
        indexer = StringIndexer(inputCol="gender", outputCol="gender_indexed")
        encoder = OneHotEncoder(inputCol="gender_indexed", outputCol="gender_encoded", dropLast=False)
        
        # VectorAssembler solo con TOP features
        assembler_selected = VectorAssembler(
            inputCols=selected_numeric_features,
            outputCol="features_vector"
        )
        
        # StandardScaler
        scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled")
        
        # Pipeline final
        pipeline_final = Pipeline(stages=[indexer, encoder, assembler_selected, scaler])
        df_final = pipeline_final.fit(df).transform(df)
        
        # Filtrar nulos SOLO en features seleccionados antes de guardar
        df_final = df_final.dropna(subset=["features_scaled"])
        
        # Guardar datos transformados
        print(f"\n[9] Guardando datos transformados...")
        df_final.select("student_id", "productivity_score", "features_scaled").write \
            .parquet(TRANSFORMED_PARQUET, mode="overwrite")
        print(f"✓ Datos transformados guardados en: {TRANSFORMED_PARQUET}")
        
        # Resumen final
        print("\n" + "="*80)
        print("  RESUMEN FINAL")
        print("="*80)
        print(f"✓ Archivos generados en: {OUT_DIR}")
        print(f"  • Correlaciones: {CORRELATIONS_CSV}")
        print(f"  • Gráfica correlaciones: {CORRELATION_PLOT}")
        print(f"  • Gráfica TOP features: {SELECTED_FEATURES_PLOT}")
        print(f"  • Reporte: {FEATURE_REPORT}")
        print(f"  • Datos transformados: {TRANSFORMED_PARQUET}")
        print(f"\n✓ Features seleccionados: {TOP_N_FEATURES}/{len(corr_list)}")
        print(f"✓ Transformaciones aplicadas: StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()
        print("Spark session cerrada.")


if __name__ == "__main__":
    main()
