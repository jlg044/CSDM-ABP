FROM python:3.11-slim

# Java (necesario para PySpark) + libs de OpenCV + procps (ps, requerido por spark-class)
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        libgl1 \
        libglib2.0-0 \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh
