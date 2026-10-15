"""Métricas a nivel de carrera. Todas promedian por carrera, no por fila.

Por qué por carrera: el campo va de 4 a 14 caballos, así que promediar por fila
le daría más peso a las carreras grandes. La pregunta de investigación es
sobre identificar al ganador de cada carrera.

Dead heats (12 carreras): top-1 cuenta acierto si el elegido es cualquiera de
los ganadores conjuntos (Sección V-E del paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _frame(df: pd.DataFrame, score) -> pd.DataFrame:
    s = df[score].to_numpy() if isinstance(score, str) else np.asarray(score)
    return pd.DataFrame({
        "race_id": df["race_id"].to_numpy(),
        "won": df["won"].to_numpy().astype(int),
        "y_rel": df["y_rel"].to_numpy() if "y_rel" in df else 0,
        "s": s.astype(float),
    })


def top1_accuracy(df, score) -> float:
    """Fracción de carreras donde el caballo con mayor score ganó.

    Con empates en el score toma el valor esperado de desempatar al azar, así
    un modelo constante da exactamente la tasa aleatoria.
    """
    f = _frame(df, score)
    is_top = f["s"] == f.groupby("race_id")["s"].transform("max")
    g = f.assign(top=is_top, hit=is_top & (f["won"] == 1)).groupby("race_id")
    return float((g["hit"].sum() / g["top"].sum()).mean())


def mrr(df, score) -> float:
    """Media del recíproco del rango del mejor ganador en el orden predicho."""
    f = _frame(df, score)
    f["rank"] = f.groupby("race_id")["s"].rank(ascending=False, method="average")
    best = f[f["won"] == 1].groupby("race_id")["rank"].min()
    return float((1.0 / best).mean())


def ndcg_at_k(df, score, k: int) -> float:
    """NDCG@k con ganancia 2^y - 1 (ecuaciones 2 y 3 del paper)."""
    f = _frame(df, score)
    gain = np.power(2.0, f["y_rel"].to_numpy()) - 1.0
    pos = f.groupby("race_id")["s"].rank(ascending=False, method="first").to_numpy()
    ideal = f.groupby("race_id")["y_rel"].rank(ascending=False, method="first").to_numpy()
    dcg = np.where(pos <= k, gain / np.log2(pos + 1), 0.0)
    idcg = np.where(ideal <= k, gain / np.log2(ideal + 1), 0.0)
    r = pd.DataFrame({"race_id": f["race_id"], "dcg": dcg, "idcg": idcg}).groupby("race_id").sum()
    r = r[r["idcg"] > 0]
    return float((r["dcg"] / r["idcg"]).mean())


def auc_within_race(df, score) -> float:
    """ROC-AUC dentro de cada carrera (ganadores vs. resto), promediado."""
    f = _frame(df, score)
    f["r"] = f.groupby("race_id")["s"].rank(ascending=True, method="average")
    g = f.groupby("race_id")
    n = g.size()
    w = g["won"].sum()
    rank_sum = f[f["won"] == 1].groupby("race_id")["r"].sum().reindex(n.index, fill_value=0)
    valid = (w > 0) & (n - w > 0)
    auc = (rank_sum - w * (w + 1) / 2) / (w * (n - w))
    return float(auc[valid].mean())


def check_probabilities(df, prob, tol: float = 1e-6) -> None:
    f = _frame(df, prob)
    if (f["s"] < 0).any() or (f["s"] > 1).any():
        raise ValueError("Probabilidades fuera de [0, 1]")
    sums = f.groupby("race_id")["s"].sum()
    if (np.abs(sums - 1) > tol).any():
        raise ValueError("Las probabilidades no suman 1 dentro de cada carrera")


def race_log_loss(df, prob) -> float:
    """-log p(ganador), promediado por carrera. Requiere p normalizada por carrera.

    Referencia: el modelo uniforme da log(n) promedio, ~2.47 con campos de 12.
    """
    f = _frame(df, prob)
    winners = f[f["won"] == 1]
    per_race = -np.log(np.clip(winners["s"], EPS, 1)).groupby(winners["race_id"]).mean()
    return float(per_race.mean())


def brier_race(df, prob) -> float:
    """Brier multiclase por carrera: sum_i (p_i - t_i)^2, con t repartido en dead heats."""
    f = _frame(df, prob)
    t = f["won"] / f.groupby("race_id")["won"].transform("sum")
    return float(((f["s"] - t) ** 2).groupby(f["race_id"]).sum().mean())


def brier_row(df, prob) -> float:
    """Brier binario por fila (la versión que suelen reportar los clasificadores)."""
    f = _frame(df, prob)
    return float(((f["s"] - f["won"]) ** 2).mean())


def reliability_table(df, prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Datos para el reliability diagram y el ECE. Cada fila es un bin."""
    f = _frame(df, prob)
    if strategy == "quantile":
        bins = pd.qcut(f["s"], q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(f["s"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    t = f.groupby(bins, observed=True).agg(p_mean=("s", "mean"), freq=("won", "mean"), n=("won", "size"))
    return t.reset_index(drop=True)


def ece(df, prob, n_bins: int = 10) -> float:
    t = reliability_table(df, prob, n_bins)
    return float((t["n"] * (t["p_mean"] - t["freq"]).abs()).sum() / t["n"].sum())


def per_race_hits(df, score) -> pd.DataFrame:
    """Por carrera: hit de top-1 (esperado si hay empates) y recíproco del rango."""
    f = _frame(df, score)
    is_top = f["s"] == f.groupby("race_id")["s"].transform("max")
    g = f.assign(top=is_top, hit=is_top & (f["won"] == 1)).groupby("race_id")
    out = pd.DataFrame({"top1": g["hit"].sum() / g["top"].sum()})
    f["rank"] = f.groupby("race_id")["s"].rank(ascending=False, method="average")
    out["rr"] = 1.0 / f[f["won"] == 1].groupby("race_id")["rank"].min()
    return out


def bootstrap_ci(df, score, ref_score=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Intervalos de confianza por bootstrap remuestreando CARRERAS (no filas).

    Con ~770 carreras por año, el error estándar de top-1 es ~1.5 puntos, así
    que una diferencia de 1-2 puntos entre modelos puede ser azar. Si se da
    `ref_score` (p. ej. el mercado), también devuelve el IC de la diferencia
    pareada: misma muestra de carreras para ambos modelos.
    """
    rng = np.random.default_rng(seed)
    a = per_race_hits(df, score)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for m in ("top1", "rr"):
        boots = a[m].to_numpy()[idx].mean(axis=1)
        name = "mrr" if m == "rr" else m
        out[f"{name}_lo"], out[f"{name}_hi"] = np.percentile(boots, [lo, hi])
    if ref_score is not None:
        b = per_race_hits(df, ref_score).reindex(a.index)
        d = (a["top1"] - b["top1"]).to_numpy()
        boots = d[idx].mean(axis=1)
        out["top1_diff_vs_ref"] = float(d.mean())
        out["top1_diff_lo"], out["top1_diff_hi"] = np.percentile(boots, [lo, hi])
    return {k: float(v) for k, v in out.items()}


def evaluate(df, score, prob=None) -> dict:
    """Todas las métricas para un modelo. `score` ordena; `prob` es calibrada."""
    out = {
        "n_races": int(df["race_id"].nunique()),
        "top1": top1_accuracy(df, score),
        "mrr": mrr(df, score),
        "ndcg@1": ndcg_at_k(df, score, 1),
        "ndcg@3": ndcg_at_k(df, score, 3),
        "ndcg@6": ndcg_at_k(df, score, 6),
        "auc_race": auc_within_race(df, score),
    }
    if prob is not None:
        check_probabilities(df, prob)
        out.update({
            "log_loss": race_log_loss(df, prob),
            "brier_race": brier_race(df, prob),
            "brier_row": brier_row(df, prob),
            "ece": ece(df, prob),
        })
    return out
