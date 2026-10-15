"""Curvas de aprendizaje y análisis de errores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .models import build


def learning_curve(name, features, params, train_df, val_df, n_years_list=None, seed=42):
    """Entrena con los últimos k años de train y mide en train y en val.

    Se usan los años MÁS RECIENTES para que la distancia temporal a val sea la
    misma en todos los puntos; solo cambia cuánta historia hay.
    Brecha train-val grande = sobreajuste; ambas bajas y planas = subajuste.
    """
    years = sorted(train_df["date"].dt.year.unique())
    n_years_list = n_years_list or list(range(1, len(years) + 1))
    rows = []
    for k in n_years_list:
        tr = train_df[train_df["date"].dt.year >= years[-k]].reset_index(drop=True)
        m = build(name, features, params, seed).fit(tr)
        for part, d in (("train", tr), ("val", val_df)):
            s = m.score(d)
            rows.append({"model": name, "years": k, "n_races": d["race_id"].nunique(), "part": part,
                         "top1": M.top1_accuracy(d, s), "mrr": M.mrr(d, s), "ndcg@3": M.ndcg_at_k(d, s, 3)})
    return pd.DataFrame(rows)


def race_table(df: pd.DataFrame, score, prob=None) -> pd.DataFrame:
    """Una fila por carrera con lo necesario para analizar aciertos y fallos."""
    f = df[["race_id", "date", "won"] + (["win_odds"] if "win_odds" in df else [])].copy()
    f["s"] = np.asarray(score)
    if prob is not None:
        f["p"] = np.asarray(prob)
    f["pred_rank"] = f.groupby("race_id")["s"].rank(ascending=False, method="first")
    if "win_odds" in f:
        f["mkt_rank"] = f.groupby("race_id")["win_odds"].rank(method="first")
    g = f.groupby("race_id")
    out = pd.DataFrame({"date": g["date"].first(), "field_size": g.size()})
    winners = f[f["won"] == 1].sort_values("pred_rank").groupby("race_id").first()
    picks = f[f["pred_rank"] == 1].set_index("race_id")
    out["hit"] = (winners["pred_rank"] == 1).reindex(out.index)
    out["winner_pred_rank"] = winners["pred_rank"]
    if prob is not None:
        out["pick_prob"] = picks["p"]
        out["winner_prob"] = winners["p"]
    if "win_odds" in f:
        out["winner_odds"] = winners["win_odds"]
        out["winner_mkt_rank"] = winners["mkt_rank"]
        out["pick_mkt_rank"] = picks["mkt_rank"]
    return out.reset_index()


def error_breakdown(rt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Top-1 segmentado: por tamaño de campo, por cuota del ganador y por si
    el modelo coincidió con el favorito del mercado."""
    out = {}
    out["by_field_size"] = (rt.assign(bucket=pd.cut(rt["field_size"], [0, 8, 11, 12, 14]))
                            .groupby("bucket", observed=True)["hit"].agg(["mean", "size"]))
    if "winner_odds" in rt:
        out["by_winner_odds"] = (rt.assign(bucket=pd.cut(rt["winner_odds"], [0, 3, 6, 10, 20, 99.1]))
                                 .groupby("bucket", observed=True)["hit"].agg(["mean", "size"]))
        out["agrees_with_market"] = (rt.assign(agrees=rt["pick_mkt_rank"] == 1)
                                     .groupby("agrees")["hit"].agg(["mean", "size"]))
    return out


def confident_misses(rt: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Los falsos positivos más caros: carreras donde el modelo estaba más
    seguro de su elegido y se equivocó. Son los casos límite a discutir."""
    if "pick_prob" not in rt:
        raise ValueError("Necesita probabilidades calibradas")
    return rt[~rt["hit"].astype(bool)].sort_values("pick_prob", ascending=False).head(n)


def missed_winners(rt: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Los falsos negativos más graves: ganadores que el modelo puso muy abajo."""
    return rt.sort_values("winner_pred_rank", ascending=False).head(n)
