"""
=============================================================================
  PIPELINE DE LIMPIEZA DE DATOS
  Dataset: student_productivity_distraction_dataset_20000.csv
  Autor: Juan (generado con Cowork / Claude)
  Fecha: 2026-04-07
=============================================================================

  ESTRATEGIA APLICADA:
  ─────────────────────────────────────────────────────────────────────────
  • Valores nulos     → IMPUTACIÓN (media o mediana según distribución)
                        · Numéricas simétricas  → media
                        · Numéricas sesgadas    → mediana
                        · Categóricas / texto   → moda (valor más frecuente)
  • Formatos de fecha → Normalización a ISO 8601 (YYYY-MM-DD)
  • Valores atípicos  → Detección con IQR + log en reporte
  • Duplicados        → Eliminación completa
  • Texto (género)    → Normalización de capitalización
  ─────────────────────────────────────────────────────────────────────────

  MODO DE USO:
      python P4-pipeline_limpieza.py

  O como módulo:
      from P4-pipeline_limpieza import limpiar_dataset
      df_limpio, reporte = limpiar_dataset("mi_archivo.csv")
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVO_ENTRADA = "student_productivity_distraction_DIRTY.csv"
ARCHIVO_SALIDA  = "student_productivity_distraction_CLEAN.csv"
ARCHIVO_REPORTE = "reporte_limpieza.txt"

# Formatos de fecha que el pipeline reconoce y unifica a YYYY-MM-DD
FORMATOS_FECHA = [
    "%d/%m/%Y",       # 31/12/2024
    "%m/%d/%Y",       # 12/31/2024
    "%d-%m-%Y",       # 31-12-2024
    "%Y/%m/%d",       # 2024/12/31
    "%Y-%m-%d",       # 2024-12-31  ← formato objetivo
    "%d %b %Y",       # 31 Dec 2024
    "%d %B %Y",       # 31 December 2024
    "%Y%m%d",         # 20241231
]

# Columnas que deben contener fechas (ajustar según el dataset)
COLUMNAS_FECHA = []   # vacío → se detectan automáticamente por nombre

# Umbral IQR para detección de outliers (1.5 = estándar; 3.0 = solo extremos)
UMBRAL_IQR = 1.5

# Umbral de asimetría (skewness) para elegir media vs mediana en imputación.
# Si |skewness| > UMBRAL_SKEW → mediana; si no → media.
UMBRAL_SKEW = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def separador(titulo: str, ancho: int = 70) -> str:
    """Devuelve una línea decorativa con título centrado."""
    return f"\n{'─' * ancho}\n  {titulo}\n{'─' * ancho}"


def parsear_fecha_flexible(valor: str) -> pd.Timestamp | None:
    """
    Intenta convertir una cadena de texto a fecha probando múltiples formatos.
    Devuelve pd.Timestamp si tiene éxito, None en caso contrario.
    """
    if pd.isna(valor) or str(valor).strip() == "":
        return None
    for fmt in FORMATOS_FECHA:
        try:
            return datetime.strptime(str(valor).strip(), fmt)
        except ValueError:
            continue
    # Último recurso: inferencia automática de pandas
    try:
        return pd.to_datetime(valor)
    except Exception:
        return None


def detectar_columnas_fecha(df: pd.DataFrame) -> list[str]:
    """
    Detecta columnas que probablemente contengan fechas basándose en:
    1. Si el nombre contiene palabras clave (date, fecha, time, hora…)
    2. Si el tipo ya es datetime
    3. Si el 80 %+ de los valores no-nulos se pueden parsear como fecha
    """
    if COLUMNAS_FECHA:                         # usar configuración manual si existe
        return [c for c in COLUMNAS_FECHA if c in df.columns]

    candidatas = []
    palabras_clave = {"date", "fecha", "time", "hora", "timestamp", "dt"}

    for col in df.columns:
        # Por nombre
        if any(kw in col.lower() for kw in palabras_clave):
            candidatas.append(col)
            continue
        # Por tipo pandas
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidatas.append(col)
            continue
        # Por contenido (solo columnas object/string)
        if df[col].dtype == object:
            muestra = df[col].dropna().head(50)
            if len(muestra) == 0:
                continue
            exitos = sum(parsear_fecha_flexible(v) is not None for v in muestra)
            if exitos / len(muestra) >= 0.8:
                candidatas.append(col)

    return candidatas


# ─────────────────────────────────────────────────────────────────────────────
# ETAPAS DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def etapa_diagnostico(df: pd.DataFrame) -> dict:
    """
    ETAPA 1 — Diagnóstico inicial.
    Calcula métricas de calidad antes de aplicar cualquier transformación.
    Devuelve un diccionario con el resumen.
    """
    filas, cols = df.shape
    nulos_por_col = df.isnull().sum()
    cols_con_nulos = nulos_por_col[nulos_por_col > 0]
    total_nulos = int(nulos_por_col.sum())
    duplicados = int(df.duplicated().sum())

    return {
        "filas_originales":   filas,
        "columnas":           cols,
        "total_nulos":        total_nulos,
        "cols_con_nulos":     cols_con_nulos,
        "filas_con_nulo":     int(df.isnull().any(axis=1).sum()),
        "duplicados":         duplicados,
        "cols_numericas":     df.select_dtypes(include="number").columns.tolist(),
        "cols_texto":         df.select_dtypes(include="object").columns.tolist(),
    }


def etapa_eliminar_duplicados(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    ETAPA 2 — Eliminar filas duplicadas exactas.
    """
    antes = len(df)
    df = df.drop_duplicates()
    eliminados = antes - len(df)
    log.append(f"  Duplicados eliminados : {eliminados:,} filas")
    return df


def etapa_normalizar_vacios(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    ETAPA 3 — Convertir cadenas vacías o solo espacios a NaN.
    Afecta únicamente a columnas de tipo object/string.
    Así la etapa de imputación las trata igual que un NaN real.
    """
    total_convertidos = 0
    detalle = []

    for col in df.select_dtypes(include="object").columns:
        mascara = df[col].astype(str).str.strip() == ""
        n = int(mascara.sum())
        if n > 0:
            df.loc[mascara, col] = np.nan
            total_convertidos += n
            detalle.append(f"    • {col:<30} {n:>5} vacíos → NaN")

    if total_convertidos > 0:
        log.append(f"  Vacíos → NaN          : {total_convertidos:,} valores convertidos")
        log.extend(detalle)
    else:
        log.append("  Vacíos → NaN          : ninguna cadena vacía detectada")

    return df


def etapa_imputar_nulos(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    ETAPA 3 — Imputar valores nulos sin eliminar filas.

    Lógica de selección automática:
    · Columnas numéricas con |skewness| ≤ UMBRAL_SKEW → media
    · Columnas numéricas con |skewness| >  UMBRAL_SKEW → mediana
    · Columnas de texto / categóricas                  → moda
    """
    nulos_totales = int(df.isnull().sum().sum())
    if nulos_totales == 0:
        log.append("  Imputación de nulos   : no hay valores nulos que imputar")
        return df

    detalle = []

    # ── Columnas numéricas ────────────────────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        n_nulos = int(df[col].isnull().sum())
        if n_nulos == 0:
            continue
        skewness = df[col].skew()
        if abs(skewness) <= UMBRAL_SKEW:
            valor = df[col].mean()
            metodo = "media"
        else:
            valor = df[col].median()
            metodo = "mediana"
        df[col] = df[col].fillna(valor)
        detalle.append(
            f"    • {col:<30} {n_nulos:>5} nulos → {metodo} ({valor:.4f})  "
            f"[skew={skewness:+.2f}]"
        )

    # ── Columnas de texto / categóricas ──────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        n_nulos = int(df[col].isnull().sum())
        if n_nulos == 0:
            continue
        moda = df[col].mode()
        valor = moda.iloc[0] if not moda.empty else "Desconocido"
        df[col] = df[col].fillna(valor)
        detalle.append(
            f"    • {col:<30} {n_nulos:>5} nulos → moda ('{valor}')"
        )

    log.append(f"  Nulos imputados       : {nulos_totales:,} valores en total")
    log.extend(detalle)
    return df


def etapa_normalizar_texto(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    ETAPA 4 — Normalizar columnas de texto.
    Aplica strip() y title-case a columnas categóricas con pocos valores únicos.
    """
    cols_texto = df.select_dtypes(include="object").columns
    normalizadas = []

    for col in cols_texto:
        n_unicos = df[col].nunique()
        if n_unicos <= 50:                     # solo columnas categóricas (pocas categorías)
            df[col] = df[col].astype(str).str.strip().str.title()
            normalizadas.append(col)

    if normalizadas:
        log.append(f"  Texto normalizado     : {normalizadas}")
    return df


def etapa_unificar_fechas(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    ETAPA 5 — Detectar columnas de fecha y unificar formato a YYYY-MM-DD.
    Las fechas que no se puedan parsear se dejan como NaT.
    """
    cols_fecha = detectar_columnas_fecha(df)

    if not cols_fecha:
        log.append("  Fechas                : ninguna columna de fecha detectada")
        return df

    convertidas = []
    no_convertidas = []

    for col in cols_fecha:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Ya es datetime → solo formatear
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
            convertidas.append(col)
        else:
            # Intentar parseo flexible fila a fila
            df[col] = df[col].apply(parsear_fecha_flexible)
            n_nat = df[col].isna().sum()
            if n_nat / len(df) < 0.5:         # al menos 50 % parseado correctamente
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
                convertidas.append(f"{col} ({n_nat} no parseados)")
            else:
                no_convertidas.append(col)

    if convertidas:
        log.append(f"  Fechas unificadas     : {convertidas}")
    if no_convertidas:
        log.append(f"  Fechas no convertidas : {no_convertidas}")

    return df


def etapa_detectar_outliers(df: pd.DataFrame, log: list) -> dict:
    """
    ETAPA 6 — Detectar outliers en columnas numéricas usando el método IQR.
    No elimina ni modifica valores; solo registra en el reporte.
    Devuelve un diccionario col → n_outliers.
    """
    cols_num = df.select_dtypes(include="number").columns
    cols_num = [c for c in cols_num if "id" not in c.lower()]  # excluir IDs

    resumen_outliers = {}
    for col in cols_num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - UMBRAL_IQR * IQR
        lim_sup = Q3 + UMBRAL_IQR * IQR
        n_out = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())
        if n_out > 0:
            resumen_outliers[col] = n_out

    if resumen_outliers:
        log.append(f"  Outliers detectados (IQR × {UMBRAL_IQR}):")
        for col, n in resumen_outliers.items():
            pct = 100 * n / len(df)
            log.append(f"    • {col:<30} {n:>5} filas  ({pct:.2f} %)")
    else:
        log.append("  Outliers              : ninguno detectado")

    return resumen_outliers


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def limpiar_dataset(
    ruta_entrada: str,
    ruta_salida: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Ejecuta el pipeline completo de limpieza sobre un archivo CSV.

    Parámetros
    ──────────
    ruta_entrada : str
        Ruta al archivo CSV de entrada.
    ruta_salida  : str | None
        Ruta donde guardar el CSV limpio. Si es None, se usa el nombre original
        con sufijo '_CLEAN'.

    Retorna
    ───────
    df_limpio : pd.DataFrame
        DataFrame limpio listo para análisis.
    reporte   : str
        Texto completo del reporte de limpieza.
    """

    inicio = datetime.now()
    log = []   # acumula mensajes del reporte

    # ── Cargar datos ──────────────────────────────────────────────────────────
    log.append(separador("CARGA DE DATOS"))
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_entrada}")

    df = pd.read_csv(ruta_entrada)
    log.append(f"  Archivo               : {ruta_entrada}")
    log.append(f"  Tamaño original       : {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # ── Diagnóstico inicial ───────────────────────────────────────────────────
    log.append(separador("DIAGNÓSTICO INICIAL"))
    diag = etapa_diagnostico(df)

    log.append(f"  Columnas numéricas    : {diag['cols_numericas']}")
    log.append(f"  Columnas de texto     : {diag['cols_texto']}")
    log.append(f"  Total valores nulos   : {diag['total_nulos']:,}")

    if diag["total_nulos"] > 0:
        log.append("  Nulos por columna:")
        for col, n in diag["cols_con_nulos"].items():
            pct = 100 * n / diag["filas_originales"]
            log.append(f"    • {col:<30} {n:>6}  ({pct:.2f} %)")
        log.append(f"  Filas afectadas       : {diag['filas_con_nulo']:,}")
    else:
        log.append("  ✓ Sin valores nulos en el dataset")

    log.append(f"  Duplicados            : {diag['duplicados']:,}")

    # ── Aplicar etapas de limpieza ────────────────────────────────────────────
    log.append(separador("TRANSFORMACIONES APLICADAS"))

    df = etapa_eliminar_duplicados(df, log)
    df = etapa_normalizar_vacios(df, log)
    df = etapa_imputar_nulos(df, log)
    df = etapa_normalizar_texto(df, log)
    df = etapa_unificar_fechas(df, log)
    outliers = etapa_detectar_outliers(df, log)

    # ── Resumen final ─────────────────────────────────────────────────────────
    log.append(separador("RESUMEN FINAL"))
    filas_finales = len(df)
    filas_eliminadas = diag["filas_originales"] - filas_finales   # solo duplicados
    nulos_imputados = int(diag["total_nulos"])
    pct_retenido = 100 * filas_finales / diag["filas_originales"]

    log.append(f"  Filas originales      : {diag['filas_originales']:>10,}")
    log.append(f"  Filas eliminadas      : {filas_eliminadas:>10,}  (solo duplicados)")
    log.append(f"  Filas finales         : {filas_finales:>10,}")
    log.append(f"  Datos retenidos       : {pct_retenido:>10.2f} %")
    log.append(f"  Valores imputados     : {nulos_imputados:>10,}  (media/mediana/moda)")
    log.append(f"  Columnas con outliers : {len(outliers):>10}")

    # ── Guardar CSV limpio ────────────────────────────────────────────────────
    if ruta_salida is None:
        base, ext = os.path.splitext(ruta_entrada)
        ruta_salida = base + "_CLEAN" + ext

    df.to_csv(ruta_salida, index=False)
    log.append(f"\n  ✓ Dataset limpio guardado en: {ruta_salida}")

    duracion = (datetime.now() - inicio).total_seconds()
    log.append(f"  ✓ Tiempo total de ejecución : {duracion:.2f} s")
    log.append("\n" + "═" * 70)

    reporte = "\n".join(log)
    return df, reporte


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Resolver rutas relativas al directorio del script
    directorio = os.path.dirname(os.path.abspath(__file__))
    entrada = os.path.join(directorio, ARCHIVO_ENTRADA)
    salida  = os.path.join(directorio, ARCHIVO_SALIDA)
    reporte_path = os.path.join(directorio, ARCHIVO_REPORTE)

    print("=" * 70)
    print("  PIPELINE DE LIMPIEZA DE DATOS")
    print("=" * 70)

    try:
        df_limpio, reporte = limpiar_dataset(entrada, salida)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # Mostrar reporte en consola
    print(reporte)

    # Guardar reporte en disco
    with open(reporte_path, "w", encoding="utf-8") as f:
        encabezado = (
            "REPORTE DE LIMPIEZA DE DATOS\n"
            f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "=" * 70 + "\n"
        )
        f.write(encabezado + reporte)

    print(f"\n  Reporte guardado en: {reporte_path}")


if __name__ == "__main__":
    main()
