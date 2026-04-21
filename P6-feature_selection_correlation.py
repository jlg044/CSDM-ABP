# P6: Ingeniería de Características para CSDM
# Mi dataset tiene 20.000 filas, uso el script para limpiar y preparar todo para ML

import os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Rutas de los archivos (mejor usar paths relativos para que el profe no tenga fallos)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "student_productivity_distraction_dataset_20000.csv")
PLOT_PATH = os.path.join(BASE_DIR, "correlacion_features.png")
SAMPLE_PATH = os.path.join(BASE_DIR, "MUESTRA_VECTORES.txt")

# Esta funcion es para ver las correlaciones en una tabla por consola
def ver_tabla_correlaciones(corrs):
    print("\nTABLA DE RELEVANCIA (Feature Selection)")
    print("-" * 50)
    # Ordeno por valor absoluto para ver las que mas influyen (da igual si es + o -)
    for nombre, valor in sorted(corrs, key=lambda x: abs(x[1]), reverse=True):
        tipo = "POSITIVO" if valor > 0 else "NEGATIVO"
        print(f"{nombre:<25} | {valor:>8.4f} | {tipo}")
    print("-" * 50 + "\n")

def main():
    # Iniciamos Spark (he bajado el log para que no salgan tantos avisos)
    spark = SparkSession.builder.appName("Practica_P6_ML").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Schema del dataset para que Spark no tenga que adivinarlo
    mi_schema = StructType([
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

    df = spark.read.schema(mi_schema).csv(INPUT_PATH, header=True)
    
    # --- PARTE 1: FEATURE SELECTION (CORRELACIONES) ---
    cols_numericas = [c.name for c in df.schema.fields if isinstance(c.dataType, (FloatType, IntegerType)) 
                      and c.name not in ["student_id", "productivity_score"]]
    
    # Calculo la correlacion de cada una con la productividad
    correlaciones = []
    for c in cols_numericas:
        res = df.stat.corr(c, "productivity_score")
        correlaciones.append((c, res))
    
    # Imprimo la tabla para la exposicion
    ver_tabla_correlaciones(correlaciones)

    # --- PARTE 2: GRAFICA DE IMPACTO ---
    # Ordeno para que la grafica de barras se vea bien
    corrs_ordenadas = sorted(correlaciones, key=lambda x: x[1])
    nombres = [x[0] for x in corrs_ordenadas]
    valores = [x[1] for x in corrs_ordenadas]
    
    plt.figure(figsize=(10, 8))
    # Ponemos rojo si es negativo y verde si es positivo (mas visual)
    colores = ['#ff4b4b' if v < 0 else '#4caf50' for v in valores]
    
    mis_barras = plt.barh(nombres, valores, color=colores)
    plt.axvline(x=0, color='black') # Linea en el cero
    
    # Añado los numeros encima de las barras para que el profe los vea
    for b in mis_barras:
        ancho = b.get_width()
        pos_x = ancho if ancho > 0 else ancho - 0.05
        plt.text(pos_x, b.get_y() + b.get_height()/2, f'{ancho:.3f}', va='center', fontweight='bold')

    plt.title('Influencia de las variables en la Productividad')
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    # --- PARTE 3: PIPELINE DE TRANSFORMACIONES ---
    # Cambio nombre a label para que Spark lo entienda mejor
    df = df.withColumnRenamed("productivity_score", "label")
    
    # Me quedo con las 8 mejores segun la tabla de antes
    correlaciones.sort(key=lambda x: abs(x[1]), reverse=True)
    mis_features = [x[0] for x in correlaciones[:8]]

    # Indexer y Encoder para el genero (categorias a numeros)
    idx = StringIndexer(inputCol="gender", outputCol="gender_idx")
    enc = OneHotEncoder(inputCol="gender_idx", outputCol="gender_vec")
    
    # Junto todo en un vector
    ass = VectorAssembler(inputCols=mis_features + ["gender_vec"], outputCol="features_sucias")
    
    # Normalizacion con StandardScaler (lo que vimos en el pizarra)
    esc = StandardScaler(inputCol="features_sucias", outputCol="features", withMean=True, withStd=True)
    
    # Monto el Pipeline completo
    mi_pipeline = Pipeline(stages=[idx, enc, ass, esc])
    df_modelo = mi_pipeline.fit(df).transform(df)
    
    # Guardo una muestra de los vectores para entregar el txt
    muestra = df_modelo.select("label", "features").limit(10).collect()
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        f.write("MUESTRA DE LOS VECTORES PARA LA ENTREGA\n\n")
        for r in muestra:
            f.write(f"Productividad: {r['label']} | Vector: {r['features']}\n")
    
    print("Script terminado. Archivos generados correctamente.")
    df_modelo.select("label", "features").show(5, truncate=False)
    
    spark.stop()

if __name__ == "__main__":
    main()
