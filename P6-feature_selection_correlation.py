"""
=============================================================================
  P6: BENCHMARK DE TIEMPO Y PRECISIÓN (R2)
  Dataset: student_productivity_distraction_dataset_20000.csv
=============================================================================
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "student_productivity_distraction_dataset_20000.csv")

def benchmark_ml(df, lista_features, nombre_test):
    """Entrena un modelo rapido y mide tiempo y precision"""
    print(f"\n>>> Test: {nombre_test}")
    inicio = time.time()
    
    # Pipeline
    idx = StringIndexer(inputCol="gender", outputCol="gender_idx")
    enc = OneHotEncoder(inputCol="gender_idx", outputCol="gender_vec")
    ass = VectorAssembler(inputCols=lista_features + ["gender_vec"], outputCol="features_raw")
    esc = StandardScaler(inputCol="features_raw", outputCol="features")
    
    # Entrenamos una Regresion Lineal rapida
    lr = LinearRegression(featuresCol="features", labelCol="label")
    
    pipe = Pipeline(stages=[idx, enc, ass, esc, lr])
    
    # Dividimos en Train y Test (80/20)
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    # Entrenar
    modelo = pipe.fit(train)
    
    # Predecir sobre test para ver precision
    predicciones = modelo.transform(test)
    
    evaluador = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = evaluador.evaluate(predicciones)
    
    fin = time.time()
    tiempo = fin - inicio
    
    print(f"✓ Tiempo: {tiempo:.4f}s | R2 (Precision): {r2:.4f}")
    return tiempo, r2

def main():
    spark = SparkSession.builder.appName("Precision_Benchmark_P6").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

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

    df = spark.read.schema(schema).csv(INPUT_PATH, header=True)
    df = df.withColumnRenamed("productivity_score", "label")
    
    # Todas las features
    todas = [c.name for c in df.schema.fields if isinstance(c.dataType, (FloatType, IntegerType)) 
             and c.name not in ["student_id", "label"]]
    
    # Top 8 (por correlacion)
    corrs = [(c, df.stat.corr(c, "label")) for c in todas]
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_8 = [x[0] for x in corrs[:8]]

    print("\n" + "="*60)
    print("      BENCHMARK DE RENDIMIENTO Y PRECISIÓN (R2)")
    print("="*60)
    
    t1, r2_todas = benchmark_ml(df, todas, "USANDO TODO")
    t2, r2_top = benchmark_ml(df, top_8, "USANDO SOLO MEJORES (Feature Selection)")
    
    print("\n" + "="*60)
    print("CONCLUSIÓN FINAL")
    print("-" * 60)
    print(f"Perdida de precision: {((r2_todas - r2_top)/r2_todas)*100:.2f}%")
    print(f"Ganancia de velocidad: {((t1 - t2)/t1)*100:.2f}%")
    print("="*60)
    
    spark.stop()

if __name__ == "__main__":
    main()
