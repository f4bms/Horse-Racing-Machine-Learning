"""Búsqueda de hiperparámetros con forward chaining dentro de train.

Folds (con train = 1997-2003):
    fold 1: entrena <= 2000, valida 2001
    fold 2: entrena <= 2001, valida 2002
    fold 3: entrena <= 2002, valida 2003
Nunca se baraja (k-fold aleatorio dejaría que el futuro informe al pasado) y
ni val (2004) ni test (2005) participan: val queda limpia para comparar
modelos y calibrar; test se toca una vez al final.

Objetivo: MRR medio en los folds. Se prefiere a top-1 porque es menos
ruidoso (top-1 es 0/1 por carrera) y premia poner al ganador arriba aunque no
quede primero.

Los baselines con hiperparámetros (logreg, gbm_binary, clogit) se ajustan con
el mismo protocolo: tunear solo el modelo propuesto inflaría su ventaja.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import metrics as M
from .models import build


def forward_chain_folds(train_df: pd.DataFrame, n_folds: int = 3):
    years = sorted(train_df["date"].dt.year.unique())
    val_years = years[-n_folds:]
    for y in val_years:
        yr = train_df["date"].dt.year
        yield y, train_df[yr < y].reset_index(drop=True), train_df[yr == y].reset_index(drop=True)


def _space(trial: optuna.Trial, name: str) -> dict:
    if name in ("lambdamart", "gbm_binary"):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "n_estimators": 2000,
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "clogit":
        return {"alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True)}
    raise ValueError(name)


def cv_score(name, features, params, train_df, n_folds=3, early_stopping=100, seed=42):
    """Devuelve dict con media y desviación estándar de MRR y top-1 en los folds,
    más la mejor iteración media (se reporta media ± desv., como en clase)."""
    mrrs, tops, iters = [], [], []
    for _, tr, va in forward_chain_folds(train_df, n_folds):
        m = build(name, features, params, seed)
        if name in ("lambdamart", "gbm_binary"):
            m.fit(tr, eval_df=va, early_stopping=early_stopping)
            iters.append(m.best_iter_)
        else:
            m.fit(tr)
        s = m.score(va)
        mrrs.append(M.mrr(va, s))
        tops.append(M.top1_accuracy(va, s))
    return {
        "mrr": float(np.mean(mrrs)), "mrr_std": float(np.std(mrrs)),
        "top1": float(np.mean(tops)), "top1_std": float(np.std(tops)),
        "best_iter": int(np.mean(iters)) if iters else None,
    }


def tune(name, features, train_df, n_trials=40, n_folds=3, seed=42, out_dir: str | Path = "results/tuning"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = _space(trial, name)
        cv = cv_score(name, features, params, train_df, n_folds, seed=seed)
        for k in ("mrr_std", "top1", "top1_std", "best_iter"):
            trial.set_user_attr(k, cv[k])
        return cv["mrr"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(out_dir / f"{name}_trials.csv", index=False)

    best = dict(study.best_params)
    it = study.best_trial.user_attrs.get("best_iter")
    if name in ("lambdamart", "gbm_binary"):
        best.update({"subsample_freq": 1, "n_estimators": int(it)})
    summary = {
        "model": name, "n_trials": n_trials, "n_folds": n_folds, "objective": "mean MRR (forward chaining)",
        "best_cv_mrr": study.best_value, "best_cv_mrr_std": study.best_trial.user_attrs["mrr_std"],
        "best_cv_top1": study.best_trial.user_attrs["top1"],
        "best_cv_top1_std": study.best_trial.user_attrs["top1_std"],
        "best_params": best, "seconds": round(elapsed, 1),
        "seconds_per_trial": round(elapsed / max(n_trials, 1), 2),
    }
    (out_dir / f"{name}_best.json").write_text(json.dumps(summary, indent=2))
    return best, summary
