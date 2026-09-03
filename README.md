# CSDM-ABP — Computación sobre Datos Masivos

Prácticas y proyecto final de la asignatura **Computación sobre Datos Masivos**, centradas en procesamiento distribuido con Apache Spark y en un pipeline completo de Machine Learning sobre un dataset de productividad estudiantil.

---

## 🎯 Proyecto final destacado: clasificador de dedos con CNN

La pieza principal del repositorio está en [`Proyecto Final/Proyecto_hands_docker`](./Proyecto%20Final/Proyecto_hands_docker): una CNN entrenada desde cero que clasifica en tiempo real cuántos dedos se muestran a la cámara, con preprocesado distribuido en PySpark, servidor de inferencia FastAPI y todo el pipeline orquestado con Docker Compose. Tiene su propio README con instrucciones detalladas.

---

## Prácticas (P2 a P7)

Serie de ejercicios progresivos sobre el mismo dataset (`student_productivity_distraction_dataset_20000.csv`), cada uno centrado en una fase distinta del ciclo de vida de los datos:

| Práctica | Contenido |
|---|---|
| **P2** | Test de estrés de un cluster Spark (`P2-spark_stress_test.py`) |
| **P3** | Comparativa de tiempos entre transformaciones y acciones en Spark, con inferencia de esquema vs `StructType` |
| **P4** | Generación de un dataset "sucio" y limpieza de datos con Spark |
| **P5** | Benchmark de formatos CSV vs Parquet |
| **P6** | Selección de features por correlación |
| **P7** | Entrenamiento y comparación de modelos de predicción (`P7-Entrenamiento_modelos.ipynb`) |

### Resultado del modelo (P7)

Se entrenaron 4 modelos para predecir la productividad de un estudiante a partir de variables como horas de estudio, asistencia, horas de sueño y nivel de estrés (80/20 train-test, validación cruzada de 5 pliegues):

| Modelo | R² | RMSE |
|---|---|---|
| **Regresión Lineal** (mejor) | 1.0000 | 0.0029 |
| Gradient Boosting | 0.9970 | 0.8791 |
| Random Forest | 0.9830 | 2.0943 |
| Árbol de Decisión | 0.7948 | 7.2698 |

Los modelos entrenados se guardan en `modelos_entrenados/` con `joblib` para poder reutilizarlos sin reentrenar.

---

## Tecnologías

- Apache Spark (PySpark) — cluster con master + 2 workers, vía Docker
- Docker Compose
- scikit-learn — entrenamiento y evaluación de modelos
- Jupyter Notebook

---

## Estructura

```
CSDM-ABP/
├── docker-compose.yml              # Cluster Spark (master + workers)
├── P2-...P7-*.py / *.ipynb         # Scripts y notebook de cada práctica
├── Entregas/                       # Informes entregados de cada práctica
├── modelos_entrenados/             # Modelos serializados (.pkl)
└── Proyecto Final/
    └── Proyecto_hands_docker/      # Proyecto destacado (ver su propio README)
```
