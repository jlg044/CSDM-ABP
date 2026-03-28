from pyspark.sql import SparkSession
import time

def main():
    # Creamos la sesion de Spark conectandonos al master de Docker
    # Aqui ajustamos la RAM y los Cores que va a usar cada executor
    spark = SparkSession.builder \
        .appName("SparkStressTest") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "512m") \
        .config("spark.executor.cores", "1") \
        .getOrCreate()

    print("SparkSession inicializada correctamente.")
    print(f"Versión de Spark: {spark.version}")

    try:
        # Generamos 10k numeros para ver como fluye la info
        # Con esto podemos monitorizar el trabajo en la Spark UI (localhost:8080)
        num_elements = 10_000
        print(f"Generando {num_elements} números y transformándolos...")
        
        # Aplicamos inmutabilidad: range_df no se toca, las transformaciones crean nuevos sets
        range_df = spark.range(0, num_elements, numPartitions=10)
        
        # Filtramos los pares (transformacion) y contamos (accion)
        # Los executors se reparten estas tareas en paralelo
        start_time = time.time()
        count = range_df.filter("id % 2 == 0").count()
        end_time = time.time()

        print(f"Resultado: Hay {count} números pares.")
        print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos.")

        # Dejamos el script corriendo para que de tiempo a ver la UI en el navegador
        print("\nRevisa la Spark UI en http://localhost:8080")
        print("Presiona Ctrl+C para finalizar la sesión.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nFinalizando sesión...")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
