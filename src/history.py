"""
Features históricas para dataset_v1: dataset_v0 + historial de caballo, jinete y entrenador.

Regla: cada feature de una fila usa solo carreras con fecha ANTERIOR a la de esa
fila. Las del mismo día tampoco cuentan (un jinete corre varias carreras por día).
Para eso los conteos se agregan por (entidad, fecha) y se desplazan una fecha.

Prior: tasa de victoria promedio de todos los caballos, asumida para cuando no se
sabe nada de el caballo.

Tasa de victoria con shrinkage:
    tasa = (victorias_previas + k * prior) / (carreras_previas + k)
Con pocas carreras la tasa se acerca al prior (tasa de victoria de train); con
muchas, a la tasa propia. k se elige por entidad en validación (nunca en test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cleaning import build_v0, validate

# Entidad -> columna que la identifica.
ENTITIES = {"horse": "horse_id", "jockey": "jockey_id", "trainer": "trainer_id"}

# Valores de k que se prueban en validación.
K_GRID = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

HISTORY_FEATURES = ["horse_win_rate", "jockey_win_rate", "trainer_win_rate",
                    "horse_days_since_last", "horse_no_history"]

# Columnas de v0 que en v1 se reemplazan (columna de v0 -> su reemplazo en v1).
# Se quitan de la tabla porque el código de modelos toma como feature toda columna
# no excluida. Para recuperarlas, v0 tiene las mismas filas con el mismo índice:
#     v0.loc[v1.index, "jockey_id"]
REPLACED = {"jockey_id": "jockey_win_rate", "trainer_id": "trainer_win_rate", "horse_gear": "gear_any"}

# Columnas que v1 agrega a v0, en orden.
NEW_FEATURES = HISTORY_FEATURES + ["gear_any"]


def previous_counts(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Carreras y victorias de `key` en fechas estrictamente anteriores a cada fila."""
    # agrupa y agrega la cantidad de carreras y victorias por key
    daily = df.groupby([key, "date"])["won"].agg(n="count", w="sum")
    # suma acumulada del día anterior por key
    prev = daily.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0)
    # devolver el resultado a cada carrera, convirtiendo el multi-indexado en columnas normales
    out = df[[key, "date"]].merge(prev.reset_index(), on=[key, "date"], how="left")
    out.index = df.index
    return out[["n", "w"]]


def shrunk_rate(counts: pd.DataFrame, k: float, prior: float) -> pd.Series:
    """Tasa con shrinkage. Sin carreras previas (n = 0) devuelve el prior."""
    n, w = counts["n"], counts["w"]
    return ((w + k * prior) / (n + k)).where(n + k > 0, prior)


def log_loss(p: pd.Series, y: pd.Series) -> float:
    """Mide que tan buena es una probabilidad, y indica si se gano (1) o no (0),
    así solo se castiga la rama"""
    # clip acota lso valores max y min posibles de la serie, incluso despues de operar
    # con esto se evita un error por log(0) = inf
    p = p.clip(1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_history(df: pd.DataFrame) -> dict:
    """Ajusta lo que el historial necesita, sin mirar test:
    - prior: tasa de victoria promedio de todos los caballos en train;
    - k por entidad: el que da menor log loss en val usando la tasa como probabilidad.
    Devuelve {"prior", "k": {entidad: k}, "search": tabla de log loss por k}."""
    prior = float(df.loc[df["split"] == "train", "won"].mean())
    val = df["split"] == "val"
    # ks guarda el mejor k de cada entidad y search la tabla con los resultados.
    ks, search = {}, []
    for name, key in ENTITIES.items():
        counts = previous_counts(df, key)
        losses = {k: log_loss(shrunk_rate(counts, k, prior)[val], df.loc[val, "won"]) for k in K_GRID}
        ks[name] = min(losses, key=losses.get) # devuelve la clave (el k) con el menor valor
        search += [{"entidad": name, "k": k, "log_loss_val": round(l, 6)} for k, l in losses.items()]
    return {"prior": prior, "k": ks, "search": pd.DataFrame(search)}


def history_features(df: pd.DataFrame, k: dict, prior: float) -> pd.DataFrame:
    """Las features históricas con k y prior ya fijados (índice igual al de `df`)."""
    out = pd.DataFrame(index=df.index)
    for name, key in ENTITIES.items():
        counts = previous_counts(df, key)
        out[f"{name}_win_rate"] = shrunk_rate(counts, k[name], prior)
        if name == "horse":
            out["horse_no_history"] = (counts["n"] == 0).astype(int)
    last = df.groupby("horse_id")["date"].shift(1) # el primer date de cada caballo queda como NaT
    out["horse_days_since_last"] = (df["date"] - last).dt.days  # NaN en el debut
    return out[HISTORY_FEATURES]


def build_v1(raw_dir: str | Path) -> tuple[pd.DataFrame, dict[str, int], dict]:
    """dataset_v0 + features históricas + gear_any, sin las columnas de REPLACED.
    Devuelve (tabla, correcciones, ajuste del historial)."""
    df, report = build_v0(raw_dir)
    fit = fit_history(df)
    df = pd.concat([df, history_features(df, fit["k"], fit["prior"])], axis=1)
    # 822 combinaciones de equipo con muy poca señal: queda solo "lleva algún equipo".
    df["gear_any"] = (df["horse_gear"] != "--").astype(int)
    df = df.drop(columns=list(REPLACED))
    validate(df)
    return df, report, fit
