"""Genera una tabla SINTÉTICA con el formato del contrato, para probar el
código de modelos mientras Persona A termina la tabla real.

NO usar estos números en el reporte: solo sirven para verificar que el
pipeline corre de punta a punta y que las métricas dan valores sensatos.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_races=6348, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(np.sort(rng.integers(
        pd.Timestamp("1997-06-02").value // 10**9, pd.Timestamp("2005-08-28").value // 10**9, n_races)), unit="s").normalize()
    sizes = rng.choice(np.arange(4, 15), size=n_races,
                       p=np.r_[[0.01] * 6, 0.03, 0.05, 0.25, 0.15, 0.46])
    rows = []
    for r in range(n_races):
        n = sizes[r]
        venue = rng.choice(["ST", "HV"], p=[0.6, 0.4])
        distance = rng.choice([1000, 1200, 1400, 1650, 1800, 2000, 2200])
        going = rng.choice(["GOOD", "GOOD TO FIRM", "GOOD TO YIELDING", "WET SLOW"])
        rating = rng.normal(60, 15, n).round()
        draw = rng.permutation(n) + 1
        weight = 113 + (rating - 60) * 0.3 + rng.normal(0, 3, n)
        no_hist = rng.random(n) < 0.055
        hist_win = np.where(no_hist, np.nan, np.clip(rng.beta(1.2, 12, n), 0, 1))
        jockey_rate = rng.beta(2, 20, n)
        country = rng.choice(["AUS", "NZ", "GB", "IRE", "USA"], n)
        draw_eff = -0.06 * draw * (1.6 if venue == "HV" else 1.0)
        strength = (0.03 * (rating - 60) + draw_eff + 4 * np.nan_to_num(hist_win, nan=0.06)
                    + 6 * jockey_rate + rng.normal(0, 0.35, n))
        perf = strength + rng.gumbel(0, 1, n)
        result = (-perf).argsort().argsort() + 1
        mkt = strength + rng.normal(0, 0.35, n)
        q = np.exp(mkt) / np.exp(mkt).sum()
        odds = np.clip(np.round(0.82 / q, 1), 1.1, 99.0)
        for i in range(n):
            rows.append({
                "race_id": r, "horse_id": rng.integers(0, 4405), "date": dates[r],
                "venue": venue, "distance": distance, "going": going,
                "horse_rating": rating[i], "draw": draw[i], "actual_weight": weight[i],
                "horse_country": country[i], "hist_win_rate": hist_win[i],
                "hist_no_history": int(no_hist[i]), "jockey_win_rate": jockey_rate[i],
                "result": result[i], "win_odds": odds[i],
            })
    df = pd.DataFrame(rows)
    df["won"] = (df["result"] == 1).astype(int)
    df["y_rel"] = np.select([df["result"] == 1, df["result"] <= 3, df["result"] <= 6], [3, 2, 1], 0)
    y = df["date"].dt.year
    df["split"] = np.where(y <= 2003, "train", np.where(y == 2004, "val", "test"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/dataset_synth.parquet")
    ap.add_argument("--n-races", type=int, default=6348)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make(a.n_races).to_parquet(a.out, index=False)
    print("escrito", a.out)
