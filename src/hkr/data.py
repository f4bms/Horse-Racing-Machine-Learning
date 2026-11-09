"""Carga de la tabla procesada que entrega Persona A y validación del contrato.

Contrato (lo mínimo que la tabla debe traer):
    race_id  : identificador de carrera
    date     : fecha de la carrera
    split    : 'train' | 'val' | 'test'
    won      : 1 si ganó (dead heats: puede haber 2 por carrera)
    y_rel    : relevancia 3/2/1/0 (ecuación 5 del paper)
    win_odds : cuota final; SOLO para el baseline de mercado, nunca como feature
Todo lo demás que no esté en RESERVED se considera feature, salvo que la
configuración dé una lista explícita.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["race_id", "date", "split", "won", "y_rel"]

# Columnas que existen en la tabla pero nunca pueden entrar al modelo.
RESERVED = {
    "race_id", "horse_id", "horse_no", "date", "split", "result", "won", "y_rel",
    "win_odds", "place_odds", "race_no",
}

# Patrones de columnas POST (Tabla IV del paper). Si alguna aparece como
# feature, es fuga: el contrato la rechaza.
FORBIDDEN_PATTERNS = [
    r"^finish_time$", r"^lengths_behind$", r"^position_sec\d", r"^behind_sec\d",
    r"^time\d", r"^sec_time\d", r"dividend", r"combination",
]

SPLITS = ("train", "val", "test")


class ContractError(ValueError):
    pass


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = sort_by_race(df)
    validate_contract(df)
    return df


def sort_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Orden cronológico y cada carrera contigua (requisito de LambdaMART)."""
    return df.sort_values(["date", "race_id"], kind="mergesort").reset_index(drop=True)


def validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ContractError(f"Faltan columnas del contrato: {missing}")

    bad_split = set(df["split"].unique()) - set(SPLITS)
    if bad_split:
        raise ContractError(f"Valores de split inválidos: {bad_split}")

    per_race = df.groupby("race_id")["split"].nunique()
    if (per_race > 1).any():
        raise ContractError(f"{(per_race > 1).sum()} carreras cruzan dos splits")

    winners = df.groupby("race_id")["won"].sum()
    if (winners < 1).any():
        raise ContractError(f"{(winners < 1).sum()} carreras sin ganador")

    # El orden cronológico de los splits: todo train antes de val, val antes de test.
    last = df.groupby("split")["date"].max()
    first = df.groupby("split")["date"].min()
    if "val" in last and last.get("train") is not None and last["train"] >= first["val"]:
        raise ContractError("Hay carreras de train posteriores al inicio de val")
    if "test" in first and "val" in last and last["val"] >= first["test"]:
        raise ContractError("Hay carreras de val posteriores al inicio de test")

    check_contiguous(df)


def check_contiguous(df: pd.DataFrame) -> None:
    rid = df["race_id"].to_numpy()
    n_blocks = 1 + int((rid[1:] != rid[:-1]).sum()) if len(rid) else 0
    if n_blocks != df["race_id"].nunique():
        raise ContractError("Las filas de una misma carrera no están contiguas")


def forbidden_columns(columns) -> list[str]:
    return [c for c in columns if any(re.search(p, c) for p in FORBIDDEN_PATTERNS)]


def feature_columns(df: pd.DataFrame, explicit: list[str] | None = None) -> list[str]:
    feats = list(explicit) if explicit else [c for c in df.columns if c not in RESERVED]
    leaks = forbidden_columns(feats) + [c for c in feats if c in RESERVED]
    if leaks:
        raise ContractError(f"Columnas prohibidas usadas como feature: {leaks}")
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ContractError(f"Features que no existen en la tabla: {missing}")
    return feats


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Vector de grupos para LambdaMART, validado contra el número de filas.

    Un vector mal armado no da error en LightGBM, simplemente entrena mal
    (riesgo 2 de la Tabla II del paper), por eso se valida aquí siempre.
    """
    check_contiguous(df)
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    sizes = np.diff(np.r_[starts, len(rid)])
    if sizes.sum() != len(df):
        raise ContractError("La suma de tamaños de grupo no coincide con las filas")
    return sizes


def race_codes(df: pd.DataFrame) -> np.ndarray:
    """Índice 0..R-1 por carrera, asumiendo filas contiguas."""
    rid = df["race_id"].to_numpy()
    return np.cumsum(np.r_[True, rid[1:] != rid[:-1]]) - 1


def split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name].reset_index(drop=True)
