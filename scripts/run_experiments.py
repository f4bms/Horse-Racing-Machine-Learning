"""Corre la escalera de modelos.

Dos etapas separadas a propósito:

  --stage dev    tunea con forward chaining dentro de train, reentrena con todo
                 train, calibra en val (2004) y reporta val. Se puede correr
                 las veces que haga falta.
  --stage final  carga los modelos ya entrenados y evalúa test (2005) UNA vez.
                 Deja un archivo candado; si existe, se niega a correr otra vez.

Uso:
  python scripts/run_experiments.py --data data/processed/dataset_v1.parquet --stage dev
  python scripts/run_experiments.py --data data/processed/dataset_v1.parquet --stage final
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hkr import analysis as A  # noqa: E402
from hkr import data as D  # noqa: E402
from hkr import metrics as M  # noqa: E402
from hkr.calibration import TemperatureCalibrator  # noqa: E402
from hkr.models import MarketBaseline, RandomBaseline, build  # noqa: E402
from hkr.tuning import tune  # noqa: E402

TUNABLE = {"logreg", "gbm_binary", "clogit", "lambdamart"}


def evaluate_split(model, cal, df, name):
    s = model.score(df)
    if name == "random":  # probabilidad exactamente uniforme, 1/n por carrera
        p = 1.0 / df.groupby("race_id")["race_id"].transform("size").to_numpy()
    else:
        p = cal.transform(s, D.race_codes(df))
    row = {"model": name, **M.evaluate(df, s, p), "T": None if name == "random" else cal.T_}
    # IC 95 % por bootstrap de carreras, y diferencia pareada contra el mercado
    ref = MarketBaseline().score(df) if ("win_odds" in df and name != "market") else None
    row.update(M.bootstrap_ci(df, s, ref))
    if name == "random":
        # top-1 e IC sobre el valor esperado (score constante = desempate al azar), no sobre una tirada
        ci0 = M.bootstrap_ci(df, np.zeros(len(df)), ref)
        row.update({k: v for k, v in ci0.items() if k.startswith("top1")})
        row["top1"] = RandomBaseline.expected_top1(df)  # valor esperado exacto, no una tirada
    return row, s, p


def stage_dev(df, cfg, out):
    seed = cfg["seed"]
    feats = D.feature_columns(df, cfg.get("features"))
    train, val = D.split(df, "train"), D.split(df, "val")
    (out / "features.json").write_text(json.dumps(feats, indent=2))
    print(f"features ({len(feats)}): {feats}")
    print(f"train {train.race_id.nunique()} carreras, val {val.race_id.nunique()}")

    rows, artifacts = [], {}
    for name in cfg["models"]:
        t0 = time.time()
        params = {}
        if name in TUNABLE:
            n_trials = cfg["tuning"]["n_trials"].get(name, 20)
            params, summ = tune(name, feats, train, n_trials, cfg["tuning"]["n_folds"], seed, out / "tuning")
            print(f"  [{name}] CV MRR={summ['best_cv_mrr']:.4f} ± {summ['best_cv_mrr_std']:.4f} "
                  f"en {summ['seconds']}s  {params}")
        model = build(name, feats, params, seed).fit(train)
        cal = TemperatureCalibrator().fit(model.score(val), D.race_codes(val), val["won"].to_numpy())
        row, s, p = evaluate_split(model, cal, val, name)
        row["fit_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        artifacts[name] = {"model": model, "calibrator": cal, "params": params}
        print(f"  [{name}] val top1={row['top1']:.4f} mrr={row['mrr']:.4f} logloss={row.get('log_loss', float('nan')):.4f}")

        if name == cfg.get("error_analysis_model"):
            rt = A.race_table(val, s, p)
            rt.to_csv(out / f"val_race_table_{name}.csv", index=False)
            for k, t in A.error_breakdown(rt).items():
                t.to_csv(out / f"val_errors_{name}_{k}.csv")
            A.confident_misses(rt).to_csv(out / f"val_confident_misses_{name}.csv", index=False)
            A.missed_winners(rt).to_csv(out / f"val_missed_winners_{name}.csv", index=False)
            M.reliability_table(val, p).to_csv(out / f"val_reliability_{name}.csv", index=False)

    for name in cfg.get("learning_curve_models", []):
        lc = A.learning_curve(name, feats, artifacts[name]["params"], train, val, seed=seed)
        lc.to_csv(out / f"learning_curve_{name}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(out / "val_metrics.csv", index=False)
    joblib.dump(artifacts, out / "models.joblib")
    (out / "run_info.json").write_text(json.dumps({
        "stage": "dev", "when": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "seed": seed, "n_rows": len(df),
    }, indent=2))
    print("\n", res[["model", "top1", "top1_lo", "top1_hi", "top1_diff_vs_ref", "top1_diff_lo", "top1_diff_hi", "mrr", "log_loss", "T"]].round(4).to_string(index=False))


def stage_final(df, cfg, out, force):
    lock = out / "TEST_EVALUATED.lock"
    if lock.exists() and not force:
        sys.exit(f"Test ya fue evaluado ({lock.read_text().strip()}). No se vuelve a tocar.")
    artifacts = joblib.load(out / "models.joblib")
    test = D.split(df, "test")
    rows = []
    for name, a in artifacts.items():
        row, s, p = evaluate_split(a["model"], a["calibrator"], test, name)
        rows.append(row)
        if name == cfg.get("error_analysis_model"):
            rt = A.race_table(test, s, p)
            rt.to_csv(out / f"test_race_table_{name}.csv", index=False)
            M.reliability_table(test, p).to_csv(out / f"test_reliability_{name}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / "test_metrics.csv", index=False)
    lit = pd.DataFrame(cfg.get("literature", []))
    lit.to_csv(out / "literature_reference.csv", index=False)
    lock.write_text(datetime.now().isoformat(timespec="seconds"))
    print(res[["model", "top1", "top1_lo", "top1_hi", "top1_diff_vs_ref", "top1_diff_lo", "top1_diff_hi", "mrr", "log_loss"]].round(4).to_string(index=False))
    print("\nReferencias de literatura:\n", lit.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/experiment.json")
    ap.add_argument("--stage", choices=["dev", "final"], required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true", help="re-evaluar test (dejar constancia en el reporte)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    np.random.seed(cfg["seed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = D.load(a.data)
    if a.stage == "dev":
        stage_dev(df, cfg, out)
    else:
        stage_final(df, cfg, out, a.force)
