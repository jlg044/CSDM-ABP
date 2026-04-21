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

def log_print(msg, file_path=None):
    """Imprime en consola y opcionalmente en archivo"""
    print(msg)
    if file_path:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")


def calculate_correlations(spark, df):
    """
    Calcula correlación de Pearson entre cada feature y la variable objetivo.
    Retorna dataframe con columnas: feature, correlacion, abs_correlacion
    """
    print("\n[*] Calculando correlaciones...")
    
    correlations_data = []
    
    # Para features numéricas: calcular correlación directa con Pandas
    pandas_df = df.select(NUMERIC_FEATURES + [TARGET]).toPandas()
    
    for feature in NUMERIC_FEATURES:
        corr = pandas_df[feature].corr(pandas_df[TARGET])
        correlations_data.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
    
    # Para features categóricas codificadas (gender_indexed)
    # La correlación se calcula después de OneHotEncoding
    corr_gender_female = pandas_df.assign(
        gender_is_female=(pandas_df['gender'] == 'Female').astype(int)
    )['gender_is_female'].corr(pandas_df[TARGET])
    
    corr_gender_male = pandas_df.assign(
        gender_is_male=(pandas_df['gender'] == 'Male').astype(int)
    )['gender_is_male'].corr(pandas_df[TARGET])
    
    correlations_data.append({
        'feature': 'gender_Female',
        'correlation': corr_gender_female,
        'abs_correlation': abs(corr_gender_female)
    })
    
    correlations_data.append({
        'feature': 'gender_Male',
        'correlation': corr_gender_male,
        'abs_correlation': abs(corr_gender_male)
    })
    
    corr_df = pd.DataFrame(correlations_data)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def select_top_features(corr_df, n=TOP_N_FEATURES):
    """Selecciona TOP N features por correlación absoluta"""
    return corr_df.head(n)


def plot_correlations(corr_df):
    """Genera gráfica de correlaciones"""
    plt.figure(figsize=(12, 8))
    
    # Preparar datos
    features_sorted = corr_df.sort_values('correlation')
    colors = ['red' if x < 0 else 'green' for x in features_sorted['correlation']]
    
    # Gráfica horizontal
    plt.barh(features_sorted['feature'], features_sorted['correlation'], color=colors, alpha=0.7)
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


def plot_top_features(top_corr_df):
    """Genera gráfica de TOP N features"""
    plt.figure(figsize=(10, 6))
    
    # Preparar datos
    features_sorted = top_corr_df.sort_values('correlation')
    colors = ['red' if x < 0 else 'green' for x in features_sorted['correlation']]
    
    # Gráfica horizontal
    plt.barh(features_sorted['feature'], features_sorted['correlation'], color=colors, alpha=0.8)
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


def generate_report(corr_df, top_corr_df):
    """Genera reporte textual de feature selection"""
    
    # Limpiar archivo anterior
    if os.path.exists(FEATURE_REPORT):
        os.remove(FEATURE_REPORT)
    
    with open(FEATURE_REPORT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE FEATURE SELECTION - CORRELACIÓN\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Variable Objetivo: {TARGET}\n")
        f.write(f"Total de features analizadas: {len(corr_df)}\n")
        f.write(f"Features seleccionados (TOP N): {TOP_N_FEATURES}\n\n")
        
        # TOP features
        f.write("-"*80 + "\n")
        f.write("TOP FEATURES SELECCIONADOS (por valor absoluto de correlación)\n")
        f.write("-"*80 + "\n\n")
        
        for idx, row in top_corr_df.iterrows():
            f.write(f"{idx+1}. {row['feature']:<30} | Correlación: {row['correlation']:>8.4f} | "
                   f"Abs: {row['abs_correlation']:>7.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("INTERPRETACIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write("• Correlación > 0: Feature aumenta con productivity_score\n")
        f.write("• Correlación < 0: Feature disminuye con productivity_score\n")
        f.write("• |Correlación| cercano a 1: Relación muy fuerte\n")
        f.write("• |Correlación| cercano a 0: Relación muy débil\n\n")
        
        # Estadísticas
        f.write("ESTADÍSTICAS DE CORRELACIÓN\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Correlación máxima (positiva):  {corr_df['correlation'].max():>8.4f} - {corr_df.loc[corr_df['correlation'].idxmax(), 'feature']}\n")
        f.write(f"Correlación mínima (negativa):  {corr_df['correlation'].min():>8.4f} - {corr_df.loc[corr_df['correlation'].idxmin(), 'feature']}\n")
        f.write(f"Correlación media (absoluta):   {corr_df['abs_correlation'].mean():>8.4f}\n")
        f.write(f"Desv. estándar (absoluta):      {corr_df['abs_correlation'].std():>8.4f}\n\n")
        
        # Features positivos vs negativos
        f.write("CLASIFICACIÓN\n")
        f.write("-"*80 + "\n\n")
        
        positive = corr_df[corr_df['correlation'] > 0]
        negative = corr_df[corr_df['correlation'] < 0]
        
        f.write(f"Features CON correlación positiva (impactan positivamente): {len(positive)}\n")
        for _, row in positive.sort_values('correlation', ascending=False).head(5).iterrows():
            f.write(f"  • {row['feature']:<30} {row['correlation']:>8.4f}\n")
        
        f.write(f"\nFeatures CON correlación negativa (impactan negativamente): {len(negative)}\n")
        for _, row in negative.sort_values('correlation').head(5).iterrows():
            f.write(f"  • {row['feature']:<30} {row['correlation']:>8.4f}\n")


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
        
        # Calcular correlaciones
        print("\n[3] Calculando correlaciones con variable objetivo...")
        corr_df = calculate_correlations(spark, df)
        
        # Guardar correlaciones a CSV
        corr_df.to_csv(CORRELATIONS_CSV, index=False)
        print(f"✓ Correlaciones guardadas en: {CORRELATIONS_CSV}")
        
        print("\nTodas las correlaciones:")
        print(corr_df.to_string())
        
        # Seleccionar TOP N
        print(f"\n[4] Seleccionando TOP {TOP_N_FEATURES} features...")
        top_corr_df = select_top_features(corr_df, n=TOP_N_FEATURES)
        
        print(f"\n✓ TOP {TOP_N_FEATURES} Features Seleccionados:")
        for idx, row in top_corr_df.iterrows():
            print(f"  {idx+1}. {row['feature']:<30} Correlación: {row['correlation']:>8.4f}")
        
        # Generar visualizaciones
        print("\n[5] Generando visualizaciones...")
        plot_correlations(corr_df)
        plot_top_features(top_corr_df)
        
        # Generar reporte
        print("\n[6] Generando reporte...")
        generate_report(corr_df, top_corr_df)
        print(f"✓ Reporte guardado en: {FEATURE_REPORT}")
        
        # TRANSFORMACIÓN: StringIndexer + OneHotEncoder para gender
        print("\n[7] Transformando features categóricas...")
        
        indexer = StringIndexer(inputCol="gender", outputCol="gender_indexed")
        encoder = OneHotEncoder(inputCol="gender_indexed", outputCol="gender_encoded")
        
        # Seleccionar features para VectorAssembler
        top_numeric_features = top_corr_df[~top_corr_df['feature'].str.startswith('gender')]['feature'].tolist()
        
        # Si gender está en TOP features, incluir gender_encoded
        has_gender = any(top_corr_df['feature'].str.startswith('gender'))
        
        if has_gender:
            assembler_inputs = top_numeric_features + ["gender_encoded"]
        else:
            assembler_inputs = top_numeric_features
        
        print(f"Features a usar en VectorAssembler: {assembler_inputs}")
        
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_vector")
        scaler = StandardScaler(inputCol="features_vector", outputCol="features_scaled")
        
        # Crear pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])
        
        # Aplicar transformaciones
        df_transformed = pipeline.fit(df).transform(df)
        
        # Guardar datos transformados
        print(f"\n[8] Guardando datos transformados...")
        df_transformed.select("student_id", "productivity_score", "features_scaled").write \
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
        print(f"\n✓ Features seleccionados: {TOP_N_FEATURES}/{len(corr_df)}")
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
