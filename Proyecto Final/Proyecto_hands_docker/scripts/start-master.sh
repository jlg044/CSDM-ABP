#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/default-java
export SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark

echo "Arrancando Spark Master en spark-master:7077 ..."
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.master.Master \
    --host spark-master \
    --port 7077 \
    --webui-port 8080
