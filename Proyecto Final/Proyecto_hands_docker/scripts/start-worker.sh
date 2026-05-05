#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/default-java
export SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

echo "Worker esperando 8s a que el master esté listo..."
sleep 8

echo "Conectando worker a spark://spark-master:7077 ..."
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
    spark://spark-master:7077
