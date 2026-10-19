"""Carga, borrado de columnas POST, limpieza, targets y partición cronológica.

Regla de admisibilidad: una variable solo puede entrar al modelo si su valor se
conocía desde la raw data. Todo lo POST se borra al leer cada archivo, ANTES de
unir races con runs, para que ninguna columna de resultado llegue a propagarse
a los corredores por el join.

Nada aquí se ajusta con estadísticas de la tabla: no se imputa, no se escala.
Los faltantes quedan como NaN y se tratan dentro del pipeline, con train.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Columnas POST, si alguna sobrevive, el código de modelos se detiene.
FORBIDDEN_PATTERNS = [
    r"^finish_time$", r"^lengths_behind$", r"^position_sec\d", r"^behind_sec\d",
    r"^time\d", r"^sec_time\d", r"dividend", r"combination",
]

REQUIRED = ["race_id", "date", "split", "won", "y_rel", "win_odds"]

# Partición por año calendario.
TRAIN_LAST_YEAR = 2003
VAL_YEAR = 2004
TEST_YEAR = 2005

# Un purasangre de carreras no pesa menos de esto (lb, peso declarado con
# equipo). Los pocos valores por debajo son errores de registro.
MIN_DECLARED_WEIGHT = 900

# Ningún campo del dataset pasa de 14 corredores. Un draw mayor que el campo
# NO es error por sí solo: si un caballo se retira, los demás conservan el
# draw asignado. Solo un draw por encima de 14 es imposible.
MAX_DRAW = 14

class DataError(ValueError):
    pass


def forbidden_columns(columns) -> list[str]:
    return [c for c in columns if any(re.search(p, c) for p in FORBIDDEN_PATTERNS)]

def drop_post(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.drop(columns=forbidden_columns(df.columns))
    left = forbidden_columns(df.columns)
    if left:
        raise DataError(f"[{name}] quedaron columnas POST: {left}")
    return df


# ---------------------------------------------------------------------------
# Paso 1: carga y borrado POST antes del join
# ---------------------------------------------------------------------------

def load_races(raw_dir: str | Path) -> pd.DataFrame:
    """races.csv sin tiempos parciales, combinaciones ni dividendos."""
    races = pd.read_csv(Path(raw_dir) / "races.csv")
    races = drop_post(races, "races.csv")
    races["date"] = pd.to_datetime(races["date"])
    return races


def load_runs(raw_dir: str | Path) -> pd.DataFrame:
    """runs.csv sin tiempos, márgenes ni posiciones parciales.
    `result` se conserva solo para construir los targets; `won` se recalcula
    desde `result` para no depender de la columna original.
    """
    runs = pd.read_csv(Path(raw_dir) / "runs.csv")
    runs = drop_post(runs, "runs.csv")
    return runs.drop(columns=["won"])


def merge(runs: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Une por race_id. El inner join descarta la carrera sin corredores."""
    for name, df in (("runs", runs), ("races", races)):
        if forbidden_columns(df.columns):
            raise DataError(f"{name} trae columnas POST antes del join")
    df = runs.merge(races, on="race_id", how="inner", validate="many_to_one")
    if len(df) != len(runs):
        raise DataError(f"El join cambió el número de corredores: {len(runs)} -> {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Paso 2: preparación (variables derivadas, revisión de rangos y tipos)
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Agrega field_size, convierte ids de jockeys y trainers en categorías y verifica rangos."""
    df = df.copy()
    report: dict[str, int] = {}

    # Cuántos caballos corrieron en esa carrera (es una variable).
    df["field_size"] = df.groupby("race_id")["horse_no"].transform("size")

    # Revisiones: no modifican nada, solo avisan si hay valores para revisar.
    checks = {
        f"declared_weight < {MIN_DECLARED_WEIGHT}": df["declared_weight"] < MIN_DECLARED_WEIGHT,
        f"draw > {MAX_DRAW}": df["draw"] > MAX_DRAW,
    }
    for nombre, bad in checks.items():
        report[nombre] = int(bad.sum())
        if bad.any():
            print(f"AVISO: Valores que se pueden revisar: {nombre} ({int(bad.sum())} filas).")

    # Identificadores: son categorías que usara el modelo, no cantidades ordenables.
    for c in ("jockey_id", "trainer_id"):
        df[c] = df[c].astype(str)

    return df, report


# ---------------------------------------------------------------------------
# Paso 3: targets, partición y orden
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """won: 1 si result == 1 (en dead heat hay dos ganadores).
    y_rel: 3 ganó, 2 llegó 2.º-3.º, 1 llegó 4.º-6.º, 0 el resto.
    """
    df = df.copy()
    r = df["result"]
    df["won"] = (r == 1).astype(int)
    df["y_rel"] = np.select([r == 1, r <= 3, r <= 6], [3, 2, 1], default=0)
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    year = df["date"].dt.year
    df["split"] = np.select(
        [year <= TRAIN_LAST_YEAR, year == VAL_YEAR, year == TEST_YEAR],
        ["train", "val", "test"], default="",
    )
    if (df["split"] == "").any():
        raise DataError(f"Años fuera de la partición: {sorted(year[df['split'] == ''].unique())}")
    return df


def sort_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Orden cronológico con las filas de cada carrera contiguas."""
    return df.sort_values(["date", "race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"Faltan columnas obligatorias: {missing}")

    post = forbidden_columns(df.columns)
    if post:
        raise DataError(f"Columnas POST en la salida: {post}")

    per_race = df.groupby("race_id")["split"].nunique() # cuenta los diferentes valores en split (si es el caso de que hay mas e uno)
    if (per_race > 1).any():
        raise DataError(f"{int((per_race > 1).sum())} carreras cruzan dos splits")

    first = df.groupby("split")["date"].min()
    last = df.groupby("split")["date"].max()
    if not (last["train"] < first["val"] and last["val"] < first["test"]):
        raise DataError("Los splits no están en orden cronológico train < val < test")

    winners = df.groupby("race_id")["won"].sum()
    if (winners < 1).any():
        raise DataError(f"{int((winners < 1).sum())} carreras sin ganador")

    rid = df["race_id"].to_numpy()
    n_blocks = 1 + int((rid[1:] != rid[:-1]).sum())
    if n_blocks != df["race_id"].nunique():
        raise DataError("Las filas de una misma carrera no están contiguas")


def build_v0(raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tabla sin historial: atributos PRE de la fila + targets + split."""
    races = load_races(raw_dir)
    runs = load_runs(raw_dir)
    df = merge(runs, races)
    df, report = prepare(df)
    df = add_targets(df)
    df = add_split(df)
    df = sort_by_race(df)

    first = ["race_id", "date", "split", "won", "y_rel", "win_odds",
             "horse_id", "horse_no", "race_no", "result", "place_odds"]
    df = df[first + [c for c in df.columns if c not in first]]
    validate(df)
    return df, report
