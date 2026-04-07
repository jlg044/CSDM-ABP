"""
=============================================================================
  GENERADOR DE DATASET SUCIO  (~7 % de datos corruptos)
  Dataset base : student_productivity_distraction_dataset_20000.csv
  Salida       : student_productivity_distraction_DIRTY.csv
  Autor        : Juan (generado con Cowork / Claude)
  Fecha        : 2026-04-07
=============================================================================

  TIPOS DE CORRUPCIÓN INTRODUCIDA:
  ─────────────────────────────────────────────────────────────────────────
  • Valores NaN en columnas numéricas        (~3.0 % de los datos)
  • Cadenas vacías / espacios en 'gender'    (~0.5 % de los datos)
  • Filas duplicadas añadidas al final       (~3.5 % de las filas)
  ─────────────────────────────────────────────────────────────────────────
  Total objetivo: ≈ 7 % del dataset corruptos
=============================================================================
"""

import pandas as pd
import numpy as np
import os

SEMILLA          = 42          # reproducibilidad
ARCHIVO_ENTRADA  = "student_productivity_distraction_dataset_20000.csv"
ARCHIVO_SALIDA   = "student_productivity_distraction_DIRTY.csv"
OBJETIVO_PCT     = 0.07        # 7 % total de corrupción


def introducir_suciedad(df: pd.DataFrame, semilla: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Corrompe un DataFrame limpio con nulos, vacíos y duplicados.
    Devuelve el DataFrame sucio y un resumen de los cambios aplicados.
    """
    rng = np.random.default_rng(semilla)
    total_celdas = df.shape[0] * df.shape[1]
    resumen = {}

    # ── 1. NaN en columnas numéricas ──────────────────────────────────────────
    # Repartimos el 3 % entre varias columnas con distintos patrones
    columnas_nulos = {
        "study_hours_per_day":    190,   # columna clave de productividad
        "sleep_hours":            150,   # hábito de sueño
        "coffee_intake_mg":       130,   # consumo de cafeína
        "exercise_minutes":       110,   # actividad física
        "attendance_percentage":   75,   # asistencia
        "final_grade":             60,   # nota final
        "productivity_score":      35,   # métrica de productividad
    }

    nulos_total = 0
    nulos_detalle = {}
    for col, n in columnas_nulos.items():
        idx = rng.choice(df.index, size=n, replace=False)
        df.loc[idx, col] = np.nan
        nulos_total += n
        nulos_detalle[col] = n

    resumen["nulos_introducidos"] = nulos_total
    resumen["nulos_detalle"]      = nulos_detalle

    # ── 2. Cadenas vacías y espacios en 'gender' ──────────────────────────────
    n_vacios  = 80
    n_espacios = 60
    idx_vacios   = rng.choice(df.index, size=n_vacios,  replace=False)
    idx_espacios = rng.choice(df.index, size=n_espacios, replace=False)

    df.loc[idx_vacios,   "gender"] = ""
    df.loc[idx_espacios, "gender"] = "   "

    resumen["vacios_introducidos"] = n_vacios + n_espacios

    # ── 3. Filas duplicadas ───────────────────────────────────────────────────
    # Tomamos filas aleatorias y las añadimos al final del DataFrame
    n_duplicados = 700
    idx_dup = rng.choice(df.index, size=n_duplicados, replace=True)
    filas_dup = df.loc[idx_dup].copy()
    df = pd.concat([df, filas_dup], ignore_index=True)

    resumen["duplicados_introducidos"] = n_duplicados

    # ── Calcular % total de corrupción (basado en filas afectadas) ───────────
    # Filas con NaN real
    mask_nulos  = df.isnull().any(axis=1)
    # Filas con cadenas vacías o solo espacios en columnas de texto
    mask_vacios = df.select_dtypes(include="object").apply(
        lambda col: col.astype(str).str.strip() == ""
    ).any(axis=1)
    filas_con_problema = int((mask_nulos | mask_vacios).sum())
    total_filas_malas  = filas_con_problema + n_duplicados
    resumen["pct_corrupcion"]      = 100 * total_filas_malas / len(df)
    resumen["filas_con_problema"]  = filas_con_problema
    resumen["filas_finales"]       = len(df)

    return df, resumen


def main():
    directorio = os.path.dirname(os.path.abspath(__file__))
    entrada    = os.path.join(directorio, ARCHIVO_ENTRADA)
    salida     = os.path.join(directorio, ARCHIVO_SALIDA)

    print("=" * 65)
    print("  GENERADOR DE DATASET SUCIO")
    print("=" * 65)

    if not os.path.exists(entrada):
        print(f"\n  ERROR: No se encontró '{ARCHIVO_ENTRADA}'")
        return

    df_original = pd.read_csv(entrada)
    print(f"\n  Dataset original  : {df_original.shape[0]:,} filas × {df_original.shape[1]} columnas")

    df_sucio, resumen = introducir_suciedad(df_original.copy(), semilla=SEMILLA)

    # ── Reporte ───────────────────────────────────────────────────────────────
    print(f"\n  ── Corrupción introducida ──────────────────────────────────")
    print(f"  NaN en columnas numéricas : {resumen['nulos_introducidos']:>5} valores")
    for col, n in resumen['nulos_detalle'].items():
        print(f"    • {col:<30} {n:>3} nulos")
    print(f"  Cadenas vacías en gender  : {resumen['vacios_introducidos']:>5} valores")
    print(f"  Filas duplicadas añadidas : {resumen['duplicados_introducidos']:>5} filas")
    print(f"\n  Dataset sucio     : {resumen['filas_finales']:,} filas × {df_sucio.shape[1]} columnas")
    print(f"  Filas con nulo/vacío      : {resumen['filas_con_problema']:,}")
    print(f"  Filas duplicadas añadidas : {resumen['duplicados_introducidos']:,}")
    print(f"  ── Corrupción total  : ≈ {resumen['pct_corrupcion']:.1f} % de las filas")

    df_sucio.to_csv(salida, index=False)
    print(f"\n  ✓ Guardado en: {salida}")
    print("=" * 65)


if __name__ == "__main__":
    main()
