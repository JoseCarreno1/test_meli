#!/usr/bin/env python3
"""
Autor: Jose carreño
Para: MeLi
Fecha: Agosto de 2025
---------------------
Genera un dataset de entrenamiento para el carrusel (MeLi).

Entradas (mismo directorio del script):
- prints.json  (JSON Lines) → impresiones del carrusel.
- taps.json    (JSON Lines) → taps o clics en el carrusel.
- pays.csv     (CSV)        → pagos realizados asociados a value_prop.

Salida:
- dataset_ready.csv → impresiones de la última semana + métricas históricas previas.
"""

import argparse
import sys
import logging
import pandas as pd
from datetime import timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------- Utilidades de validación --------
def _validar_columnas(df: pd.DataFrame, requeridas: list, nombre: str) -> None:
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {faltantes}")

def _asegurar_archivo(ruta: Path) -> None:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {ruta}")

# -------- Carga y normalización --------
def cargar_jsonl(ruta: Path) -> pd.DataFrame:
    """Carga un archivo JSON Lines en un DataFrame, con manejo de errores."""
    _asegurar_archivo(ruta)
    try:
        df = pd.read_json(ruta, lines=True)
    except ValueError as e:
        raise ValueError(f"Error al parsear JSON Lines en {ruta}: {e}") from e
    if df.empty:
        logging.warning(f"Archivo vacío: {ruta}")
    return df

def normalizar_eventos(df: pd.DataFrame, col_dia: str = "day") -> pd.DataFrame:
    """
    Convierte el JSON de eventos en un DataFrame con columnas planas.
    Requiere: 'user_id', 'event_data' (dict con 'value_prop' y 'position').
    """
    _validar_columnas(df, [col_dia, "user_id", "event_data"], "eventos")
    # Validar que event_data sea dict-like
    if not df["event_data"].apply(lambda x: isinstance(x, dict)).all():
        raise TypeError("La columna 'event_data' debe contener diccionarios por fila.")
    try:
        out = pd.DataFrame({
            "day": pd.to_datetime(df[col_dia], errors="coerce"),
            "user_id": df["user_id"],
            "value_prop": df["event_data"].apply(lambda x: x.get("value_prop")),
            "position": df["event_data"].apply(lambda x: x.get("position")),
        })
    except Exception as e:
        raise RuntimeError(f"Error al normalizar eventos: {e}") from e

    # Validaciones de nulos críticos
    if out["day"].isna().any():
        raise ValueError("Existen fechas inválidas en 'day' tras la conversión a datetime.")
    if out["user_id"].isna().any():
        raise ValueError("Existen 'user_id' nulos tras la normalización de eventos.")
    return out

# -------- Agregaciones --------
def conteos_diarios(df: pd.DataFrame, nombre_columna: str) -> pd.DataFrame:
    """Conteo diario por (user_id, value_prop, day). Si df está vacío, retorna estructura vacía."""
    if df.empty:
        return pd.DataFrame(columns=["user_id", "value_prop", "day", nombre_columna])
    g = df.groupby(["user_id", "value_prop", "day"]).size().rename(nombre_columna).reset_index()
    return g

def ventana_prev_21d(df: pd.DataFrame, col_metrica: str) -> pd.DataFrame:
    """
    Suma de los últimos 21 días excluyendo el día actual, por (user_id, value_prop).
    Retorna DataFrame con columnas: ['day', f'{col_metrica}_prev_3w', 'user_id', 'value_prop'].
    """
    # Estructura vacía si no hay datos
    if df.empty:
        return pd.DataFrame(columns=["day", f"{col_metrica}_prev_3w", "user_id", "value_prop"])

    _validar_columnas(df, ["user_id", "value_prop", "day", col_metrica], "ventana_prev_21d")
    df = df.copy()
    resultados = []
    try:
        for (uid, vp), grupo in df.groupby(["user_id", "value_prop"], sort=False):
            if grupo.empty:
                continue
            # Rango continuo para incluir días sin eventos
            idx = pd.date_range(grupo["day"].min(), grupo["day"].max(), freq="D")
            ts = grupo.set_index("day")[col_metrica].reindex(idx, fill_value=0).sort_index()
            prev = ts.rolling(window=21, min_periods=1).sum().shift(1).fillna(0)
            salida = prev.rename(f"{col_metrica}_prev_3w").reset_index().rename(columns={"index": "day"})
            salida["user_id"] = uid
            salida["value_prop"] = vp
            resultados.append(salida)
    except Exception as e:
        raise RuntimeError(f"Error calculando ventana móvil 21d para '{col_metrica}': {e}") from e

    if not resultados:
        return pd.DataFrame(columns=["day", f"{col_metrica}_prev_3w", "user_id", "value_prop"])
    return pd.concat(resultados, ignore_index=True)

# -------- Construcción del dataset --------
def construir_dataset(dir_entrada: Path, salida_csv: Path) -> None:
    """Construye el dataset final según las reglas definidas (con validaciones y errores controlados)."""

    # Validar directorio de entrada
    if not dir_entrada.exists() or not dir_entrada.is_dir():
        raise NotADirectoryError(f"Directorio de entrada inválido: {dir_entrada}")

    # Cargar datos de entrada
    prints_raw = cargar_jsonl(dir_entrada / "prints.json")
    taps_raw   = cargar_jsonl(dir_entrada / "taps.json")
    _asegurar_archivo(dir_entrada / "pays.csv")
    try:
        pays_raw   = pd.read_csv(dir_entrada / "pays.csv")
    except Exception as e:
        raise ValueError(f"Error al leer pays.csv: {e}") from e

    # Normalizar estructuras
    prints = normalizar_eventos(prints_raw, "day")
    taps   = normalizar_eventos(taps_raw, "day")

    # Validar columnas en pays
    _validar_columnas(pays_raw, ["pay_date", "user_id", "value_prop", "total"], "pays.csv")
    pays = pd.DataFrame({
        "day": pd.to_datetime(pays_raw["pay_date"], errors="coerce"),
        "user_id": pays_raw["user_id"],
        "value_prop": pays_raw["value_prop"],
        "amount": pd.to_numeric(pays_raw["total"], errors="coerce")
    })
    if pays["day"].isna().any():
        raise ValueError("Existen fechas inválidas en 'pay_date' de pays.csv")
    if pays["amount"].isna().any():
        raise ValueError("Existen montos no numéricos en 'total' de pays.csv")

    if prints.empty:
        # No hay impresiones → exportar CSV vacío con cabeceras esperadas
        logging.warning("No se encontraron impresiones en 'prints'. Se generará un CSV vacío con cabeceras.")
        columnas = ["day", "user_id", "value_prop", "position", "clicked",
                    "prints_prev_3w", "taps_prev_3w", "payments_prev_3w", "amount_prev_3w"]
        pd.DataFrame(columns=columnas).to_csv(salida_csv, index=False)
        return

    # Determinar última semana según fecha máxima en prints
    max_day = prints["day"].max()
    if pd.isna(max_day):
        raise ValueError("No fue posible determinar 'max_day' en prints (fechas inválidas).")
    inicio_semana = max_day - timedelta(days=6)
    prints_semana = prints[(prints["day"] >= inicio_semana) & (prints["day"] <= max_day)].copy()
    if prints_semana.empty:
        logging.warning("No hay impresiones en la última semana. Se generará CSV vacío con cabeceras.")
        columnas = ["day", "user_id", "value_prop", "position", "clicked",
                    "prints_prev_3w", "taps_prev_3w", "payments_prev_3w", "amount_prev_3w"]
        pd.DataFrame(columns=columnas).to_csv(salida_csv, index=False)
        return

    # Agregaciones diarias
    prints_diarios = conteos_diarios(prints, "prints")
    taps_diarios   = conteos_diarios(taps, "taps")
    # pays: conteo y monto por día
    if pays.empty:
        logging.warning("El archivo de pagos está vacío; se usarán ceros para métricas de pagos.")
        pays_diarios = pd.DataFrame(columns=["user_id", "value_prop", "day", "payments", "amount"])
    else:
        pays_cnt = pays.groupby(["user_id", "value_prop", "day"]).size().rename("payments").reset_index()
        pays_amt = pays.groupby(["user_id", "value_prop", "day"])["amount"].sum().rename("amount").reset_index()
        pays_diarios = pd.merge(pays_cnt, pays_amt, on=["user_id", "value_prop", "day"], how="outer").fillna(0)

    # Cálculo de ventanas previas (21 días)
    roll_prints = ventana_prev_21d(prints_diarios, "prints")
    roll_taps   = ventana_prev_21d(taps_diarios, "taps")
    roll_pagos  = ventana_prev_21d(pays_diarios, "payments")
    roll_montos = ventana_prev_21d(pays_diarios, "amount")

    # Base: prints de la última semana
    features = prints_semana[["day", "user_id", "value_prop", "position"]].copy()

    # Marcar si hubo click el mismo día para la misma value_prop
    if not taps_diarios.empty:
        taps_diarios = taps_diarios.copy()
        taps_diarios["clicked"] = (taps_diarios["taps"] > 0).astype(int)
        features = features.merge(
            taps_diarios[["day", "user_id", "value_prop", "clicked"]],
            on=["day", "user_id", "value_prop"], how="left"
        )
        features["clicked"] = features["clicked"].fillna(0).astype(int)
    else:
        features["clicked"] = 0

    # Unir métricas históricas (si están vacías, no rompen el merge)
    for df_roll in [
        (roll_prints, ["day", "user_id", "value_prop", "prints_prev_3w"]),
        (roll_taps,   ["day", "user_id", "value_prop", "taps_prev_3w"]),
        (roll_pagos,  ["day", "user_id", "value_prop", "payments_prev_3w"]),
        (roll_montos, ["day", "user_id", "value_prop", "amount_prev_3w"]),
    ]:
        roll_df, cols = df_roll
        if not roll_df.empty:
            # Asegurar nombres esperados
            if cols[-1] not in roll_df.columns:
                # renombrar última col calculada al nombre esperado
                calc_col = [c for c in roll_df.columns if c.endswith("_prev_3w")][-1]
                roll_df = roll_df.rename(columns={calc_col: cols[-1]})
            features = features.merge(roll_df[cols], on=["day", "user_id", "value_prop"], how="left")

    # Llenar métricas históricas faltantes con 0
    for c in ["prints_prev_3w", "taps_prev_3w", "payments_prev_3w", "amount_prev_3w"]:
        if c not in features.columns:
            features[c] = 0
        else:
            features[c] = features[c].fillna(0)

    # Ordenar y exportar
    features = features.sort_values(["day", "user_id", "value_prop", "position"]).reset_index(drop=True)
    try:
        features.to_csv(salida_csv, index=False)
    except PermissionError as e:
        raise PermissionError(f"No se pudo escribir la salida en {salida_csv}: {e}") from e

def main():
    parser = argparse.ArgumentParser(description="Generar dataset Carrusel XSelling (MeLi)")
    parser.add_argument("--input_dir", type=Path, default=Path("."), help="Directorio con los archivos de entrada")
    parser.add_argument("--output", type=Path, default=Path("dataset_ready.csv"), help="Ruta de salida del CSV")
    args = parser.parse_args()

    try:
        construir_dataset(args.input_dir, args.output)
        logging.info(f"Proceso finalizado correctamente. Salida: {args.output}")
    except Exception as e:
        logging.error(f"Fallo del proceso: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
