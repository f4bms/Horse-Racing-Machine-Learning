"""La escalera de modelos (Sección IV-E del paper), de menor a mayor complejidad.

Todos comparten la misma interfaz:
    m.fit(train_df)          -> ajusta con train, nada más
    m.score(df) -> ndarray   -> score para ordenar (mayor = más probable ganar)

| Escalón             | Qué aísla                                              |
|---------------------|--------------------------------------------------------|
| random              | piso: ¿el modelo aprende algo?                         |
| market              | referencia externa (cuotas finales), no es feature     |
| logreg              | baseline razonable: gana/no gana, fila por fila        |
| gbm_binary          | lo mismo con no linealidad (el enfoque dominante)      |
| clogit              | + respeta que las probabilidades de la carrera suman 1 |
| lambdamart          | + aprende del orden completo de llegada                |
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import data as D
from .preprocess import TreeCategoricals, linear_preprocessor, split_types

EPS = 1e-12
# LightGBM >= 4.6 avisa que eval_set cambiará de nombre; seguimos usándolo para
# ser compatibles con versiones anteriores.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


class RandomBaseline:
    """Score uniforme. Se reporta además el valor esperado exacto de top-1."""
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train_df):
        return self

    def score(self, df):
        return np.random.default_rng(self.seed).random(len(df))

    @staticmethod
    def expected_top1(df) -> float:
        g = df.groupby("race_id")
        return float((g["won"].sum() / g.size()).mean())


class MarketBaseline:
    """Favorito del mercado: score = log(1 / win_odds).

    Ojo: win_odds está topada en 99 (9.4 % de los caballos), así que el orden
    entre los más largos es arbitrario. No afecta top-1 en la práctica, pero
    sí la calibración de los outsiders.
    """
    name = "market"

    def fit(self, train_df):
        return self

    def score(self, df):
        return np.log(1.0 / df["win_odds"].to_numpy())


class _FeatureModel:
    def __init__(self, features: list[str]):
        self.features = features


class LogRegBaseline(_FeatureModel):
    """Regresión logística regularizada (L2). La regularización también resuelve
    la colinealidad horse_rating / actual_weight (Tabla V del paper).
    Sin class_weight: la probabilidad se recalibra después dentro de cada carrera."""
    name = "logreg"

    def __init__(self, features, C: float = 1.0):
        super().__init__(features)
        self.C = C

    def fit(self, train_df):
        num, cat = split_types(train_df, self.features)
        self.pipe_ = Pipeline([
            ("prep", linear_preprocessor(num, cat)),
            ("clf", LogisticRegression(C=self.C, max_iter=3000)),
        ])
        self.pipe_.fit(train_df[self.features], train_df["won"])
        return self

    def score(self, df):
        return np.log(np.clip(self.pipe_.predict_proba(df[self.features])[:, 1], EPS, 1))


class GBMBinary(_FeatureModel):
    """Gradient boosting binario: el formato dominante de la literatura (Eje A)."""
    name = "gbm_binary"

    def __init__(self, features, params: dict | None = None, seed: int = 42):
        super().__init__(features)
        self.params = params or {}
        self.seed = seed

    def fit(self, train_df, eval_df=None, early_stopping: int | None = None):
        _, cat = split_types(train_df, self.features)
        self.enc_ = TreeCategoricals(cat).fit(train_df[self.features])
        p = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
             "verbose": -1, "random_state": self.seed, **self.params}
        self.clf_ = LGBMClassifier(**p)
        kw = {}
        if eval_df is not None and early_stopping:
            from lightgbm import early_stopping as es
            kw = {"eval_set": [(self.enc_.transform(eval_df[self.features]), eval_df["won"])],
                  "callbacks": [es(early_stopping, verbose=False)]}
        self.clf_.fit(self.enc_.transform(train_df[self.features]), train_df["won"], **kw)
        self.best_iter_ = self.clf_.best_iteration_ or p["n_estimators"]
        return self

    def score(self, df):
        p = self.clf_.predict_proba(self.enc_.transform(df[self.features]))[:, 1]
        return np.log(np.clip(p, EPS, 1))


class ConditionalLogit(_FeatureModel):
    """Logit condicional (McFadden) con penalización L2.

    P(i gana | carrera r) = exp(x_i·b) / sum_{j en r} exp(x_j·b)

    La diferencia con la logística: la normalización ocurre dentro de la
    carrera, así que el modelo aprende de la comparación entre rivales y las
    probabilidades suman 1 por construcción. No hay intercepto (se cancela).
    """
    name = "clogit"

    def __init__(self, features, alpha: float = 1e-3):
        super().__init__(features)
        self.alpha = alpha

    @staticmethod
    def _loss_grad(b, X, codes, starts, t, n_races, alpha):
        u = X @ b
        m = np.maximum.reduceat(u, starts)
        e = np.exp(u - m[codes])
        s = np.add.reduceat(e, starts)
        lse = m + np.log(s)
        p = e / s[codes]
        nll = -(t @ u - lse.sum()) / n_races
        grad = X.T @ (p - t) / n_races
        return nll + alpha * b @ b, grad + 2 * alpha * b

    def fit(self, train_df):
        num, cat = split_types(train_df, self.features)
        self.prep_ = linear_preprocessor(num, cat).fit(train_df[self.features])
        X = np.asarray(self.prep_.transform(train_df[self.features]), dtype=float)
        codes = D.race_codes(train_df)
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        won = train_df["won"].to_numpy().astype(float)
        t = won / np.bincount(codes, weights=won)[codes]  # dead heats: reparte la masa
        res = minimize(self._loss_grad, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       args=(X, codes, starts, t, len(starts), self.alpha),
                       options={"maxiter": 2000})
        self.coef_ = res.x
        self.converged_ = bool(res.success)
        self.feature_names_ = list(self.prep_.get_feature_names_out())
        return self

    def score(self, df):
        return np.asarray(self.prep_.transform(df[self.features]), dtype=float) @ self.coef_


class LambdaMART(_FeatureModel):
    """Modelo propuesto: LGBMRanker con objetivo lambdarank.

    Etiqueta: y_rel (3/2/1/0). El label_gain por defecto de LightGBM es 2^y - 1,
    el mismo de la ecuación 2 del paper.
    """
    name = "lambdamart"

    def __init__(self, features, params: dict | None = None, seed: int = 42):
        super().__init__(features)
        self.params = params or {}
        self.seed = seed

    def fit(self, train_df, eval_df=None, early_stopping: int | None = None):
        _, cat = split_types(train_df, self.features)
        self.enc_ = TreeCategoricals(cat).fit(train_df[self.features])
        p = {"objective": "lambdarank", "n_estimators": 500, "learning_rate": 0.05,
             "num_leaves": 31, "verbose": -1, "random_state": self.seed, **self.params}
        self.model_ = LGBMRanker(**p)
        kw = {}
        if eval_df is not None and early_stopping:
            from lightgbm import early_stopping as es
            kw = {"eval_set": [(self.enc_.transform(eval_df[self.features]), eval_df["y_rel"])],
                  "eval_group": [D.group_sizes(eval_df)], "eval_at": [1, 3],
                  "callbacks": [es(early_stopping, first_metric_only=True, verbose=False)]}
        self.model_.fit(self.enc_.transform(train_df[self.features]), train_df["y_rel"],
                        group=D.group_sizes(train_df), **kw)
        self.best_iter_ = self.model_.best_iteration_ or p["n_estimators"]
        return self

    def score(self, df):
        return self.model_.predict(self.enc_.transform(df[self.features]))


def build(name: str, features: list[str], params: dict | None = None, seed: int = 42):
    params = dict(params or {})
    if name == "random":
        return RandomBaseline(seed)
    if name == "market":
        return MarketBaseline()
    if name == "logreg":
        return LogRegBaseline(features, **params)
    if name == "gbm_binary":
        return GBMBinary(features, params, seed)
    if name == "clogit":
        return ConditionalLogit(features, **params)
    if name == "lambdamart":
        return LambdaMART(features, params, seed)
    raise ValueError(name)
