"""De scores a probabilidades de ganar que suman 1 en cada carrera.

Todos los modelos producen un score (LambdaMART: score sin escala; clasificadores:
log p; mercado: log de la probabilidad implícita). La probabilidad es

    p_i = exp(s_i / T) / sum_{j en la carrera} exp(s_j / T)

y T se ajusta minimizando el log loss por carrera SOLO en validación (2004).
Con un solo parámetro no cambia el orden, así que top-1, MRR y NDCG no se
mueven; solo cambian log loss, Brier y ECE.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def race_softmax(scores: np.ndarray, codes: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / T
    n = codes.max() + 1
    zmax = np.full(n, -np.inf)
    np.maximum.at(zmax, codes, z)
    e = np.exp(z - zmax[codes])
    return e / np.bincount(codes, weights=e, minlength=n)[codes]


def race_nll(scores, codes, won, T: float) -> float:
    p = race_softmax(scores, codes, T)
    w = won.astype(bool)
    n_win = np.bincount(codes, weights=won, minlength=codes.max() + 1)
    per_row = -np.log(np.clip(p[w], 1e-12, 1)) / n_win[codes[w]]
    return per_row.sum() / len(n_win)


class TemperatureCalibrator:
    def __init__(self, bounds=(0.05, 20.0)):
        self.bounds = bounds
        self.T_ = 1.0

    def fit(self, scores, codes, won):
        res = minimize_scalar(
            lambda logT: race_nll(scores, codes, won, np.exp(logT)),
            bounds=np.log(self.bounds), method="bounded",
        )
        self.T_ = float(np.exp(res.x))
        return self

    def transform(self, scores, codes):
        return race_softmax(scores, codes, self.T_)
