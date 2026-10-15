import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hkr import data as D  # noqa: E402
from hkr import metrics as M  # noqa: E402
from hkr.calibration import TemperatureCalibrator, race_softmax  # noqa: E402
from hkr.models import ConditionalLogit  # noqa: E402


def two_races():
    # carrera 1: 4 caballos, ganó el 0. carrera 2: 3 caballos, ganó el 2.
    return pd.DataFrame({
        "race_id": [1, 1, 1, 1, 2, 2, 2],
        "won":     [1, 0, 0, 0, 0, 0, 1],
        "y_rel":   [3, 2, 2, 1, 2, 1, 3],
    })


def test_perfect_and_reversed():
    df = two_races()
    perfect = np.array([4, 3, 2, 1, 2, 1, 3], float)
    assert M.top1_accuracy(df, perfect) == 1
    assert M.mrr(df, perfect) == 1
    assert M.auc_within_race(df, perfect) == 1
    assert M.ndcg_at_k(df, perfect, 3) == pytest.approx(1.0)
    rev = -perfect
    assert M.top1_accuracy(df, rev) == 0
    assert M.mrr(df, rev) == pytest.approx((1 / 4 + 1 / 3) / 2)
    assert M.auc_within_race(df, rev) == 0


def test_constant_score_gives_random_rate():
    df = two_races()
    assert M.top1_accuracy(df, np.zeros(7)) == pytest.approx((1 / 4 + 1 / 3) / 2)
    assert M.auc_within_race(df, np.zeros(7)) == pytest.approx(0.5)


def test_dead_heat_counts_either_winner():
    df = pd.DataFrame({"race_id": [1, 1, 1], "won": [1, 1, 0], "y_rel": [3, 3, 0]})
    assert M.top1_accuracy(df, np.array([1, 2, 0.0])) == 1
    assert M.mrr(df, np.array([0, 2, 1.0])) == 1


def test_softmax_sums_to_one_per_race():
    codes = np.array([0, 0, 0, 1, 1])
    p = race_softmax(np.array([1, 2, 3, 0.5, 0.5]), codes, T=0.7)
    assert np.bincount(codes, weights=p) == pytest.approx([1, 1])


def test_temperature_preserves_order_and_recovers_scale():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(3000), 10)
    s_true = rng.normal(0, 1, len(codes))
    # ganador muestreado con softmax(s_true): el score observado está escalado x3
    won = np.zeros(len(codes))
    for r in range(3000):
        idx = np.arange(r * 10, r * 10 + 10)
        p = np.exp(s_true[idx]) / np.exp(s_true[idx]).sum()
        won[idx[rng.choice(10, p=p)]] = 1
    cal = TemperatureCalibrator().fit(3 * s_true, codes, won)
    assert cal.T_ == pytest.approx(3, rel=0.15)


def test_clogit_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3))
    codes = np.repeat([0, 1, 2], 4)
    starts = np.array([0, 4, 8])
    t = np.zeros(12)
    t[[1, 6, 8]] = 1
    b = rng.normal(size=3)
    f, g = ConditionalLogit._loss_grad(b, X, codes, starts, t, 3, 0.1)
    num = np.array([(ConditionalLogit._loss_grad(b + e, X, codes, starts, t, 3, 0.1)[0] - f) / 1e-6
                    for e in np.eye(3) * 1e-6])
    assert g == pytest.approx(num, abs=1e-4)


def test_group_sizes_and_contiguity():
    df = pd.DataFrame({"race_id": [5, 5, 7, 7, 7]})
    assert list(D.group_sizes(df)) == [2, 3]
    with pytest.raises(D.ContractError):
        D.group_sizes(pd.DataFrame({"race_id": [5, 7, 5]}))


def test_contract_rejects_post_race_features():
    with pytest.raises(D.ContractError):
        D.feature_columns(pd.DataFrame(columns=["draw", "finish_time"]), ["draw", "finish_time"])
    with pytest.raises(D.ContractError):
        D.feature_columns(pd.DataFrame(columns=["draw", "win_odds"]), ["draw", "win_odds"])
    inferred = D.feature_columns(pd.DataFrame(columns=["race_id", "won", "draw", "win_odds"]))
    assert inferred == ["draw"]


def test_contract_rejects_race_crossing_splits():
    df = pd.DataFrame({"race_id": [1, 1], "date": pd.to_datetime(["2003-12-31"] * 2),
                       "split": ["train", "val"], "won": [1, 0], "y_rel": [3, 0]})
    with pytest.raises(D.ContractError):
        D.validate_contract(df)


def test_bootstrap_ci_contains_point_and_paired_diff_zero_for_same_model():
    rng = np.random.default_rng(3)
    n_r, n = 300, 10
    df = pd.DataFrame({"race_id": np.repeat(np.arange(n_r), n), "won": 0, "y_rel": 0})
    df.loc[np.arange(n_r) * n + rng.integers(0, n, n_r), "won"] = 1
    s = rng.random(len(df)) + df["won"] * 0.5
    ci = M.bootstrap_ci(df, s, ref_score=s, n_boot=500)
    t = M.top1_accuracy(df, s)
    assert ci["top1_lo"] <= t <= ci["top1_hi"]
    assert ci["top1_diff_vs_ref"] == 0 and ci["top1_diff_lo"] == 0 and ci["top1_diff_hi"] == 0
