"""
Tests contra fuga de datos sobre la tabla real dataset_v0.

Cubren las tres formas de fuga sugeridas:
  - target leakage: columnas que solo se conocen después de la carrera;
  - fuga temporal: carreras futuras mezcladas en train o partidas entre splits;
  - fuga por preprocesamiento: transformaciones ajustadas con toda la tabla (imputar, escalar) antes de particionar.

Las justificaciones de cada regla están en docs/particion_y_fuga.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaning import (
    MAX_DRAW, MIN_DECLARED_WEIGHT, TRAIN_LAST_YEAR, VAL_YEAR, TEST_YEAR, forbidden_columns,
)

from data_dictionary import DESCRIPCIONES, NON_FEATURES, feature_columns

# Columnas POST de cada CSV raw. La lista es explícita a propósito: si el dataset cambia o un patrón de
# FORBIDDEN_PATTERNS deja de reconocer alguna, el test falla y obliga a revisar.
POST_RUNS = (
    ["lengths_behind"]
    + [f"position_sec{i}" for i in range(1, 7)]
    + [f"behind_sec{i}" for i in range(1, 7)]
    + [f"time{i}" for i in range(1, 7)]
    + ["finish_time"]
)
POST_RACES = (
    [f"sec_time{i}" for i in range(1, 8)]
    + [f"time{i}" for i in range(1, 8)]
    + [f"place_combination{i}" for i in range(1, 5)]
    + [f"place_dividend{i}" for i in range(1, 5)]
    + ["win_combination1", "win_dividend1", "win_combination2", "win_dividend2"]
)

# Features aprobadas para v0, revisadas una por una en docs/particion_y_fuga.md.
# Agregar una feature exige agregarla aquí, es decir, revisarla contra fuga.
EXPECTED_FEATURES_V0 = [
    "horse_age", "horse_country", "horse_type", "horse_rating", "horse_gear",
    "declared_weight", "actual_weight", "draw", "trainer_id", "jockey_id",
    "venue", "config", "surface", "distance", "going", "horse_ratings",
    "prize", "race_class", "field_size",
]

# Columnas que v0 crea y que no existen en los CSV raw. `won` sí existe en
# runs.csv: se descarta y se recalcula desde `result` (ver test_won_recalculado_coincide_con_el_original).
DERIVED = {"split", "y_rel", "field_size"}

# Columnas que describen la carrera y deben ser iguales para todos sus corredores.
RACE_LEVEL = ["date", "venue", "race_no", "config", "surface", "distance", "going",
              "horse_ratings", "prize", "race_class", "field_size", "split"]

# Umbrales para detectar una feature que "sabe" el resultado. win_odds (cuota final)
# es la referencia: debe superarlos; ninguna feature debería acercarse a ella.
MAX_RHO = 0.3       # numéricas, dentro de la carrera. v0: máx draw 0.105, win_odds 0.494
MAX_AUC_GAP = 0.25  # categóricas, |AUC - 0.5|.    v0: máx jockey_id 0.649, win_odds 0.80
# Aun así una variable legítima puede estar muy relacionada con el resultado, y una variable con fuga
# podría tener una correlación baja, por eso es una heuristica

def _rho_en_carrera(df, col):
    """Spearman medio entre `col` y `result` dentro de cada carrera (NaN si `col`
    es igual para todos los corredores, como distance o prize)."""
    rangos = df[[col, "result"]].astype(float).groupby(df["race_id"]).rank()
    return rangos.groupby(df["race_id"]).corr().xs(col, level=1)["result"].mean()


# Spearman evalúa la fuerza y dirección de la asociación monotónica entre dos variables.
def _auc_en_val(df, col):
    """AUC en val de `col` sola, reemplazando cada valor por su tasa de victoria en train."""
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    tasa = train.groupby(col)["won"].mean()
    score = val[col].map(tasa).fillna(train["won"].mean()).rank()
    y = val["won"].to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return (score[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _numericas(df):
    return [c for c in feature_columns(df.columns) if pd.api.types.is_numeric_dtype(df[c])]


def _categoricas(df):
    return [c for c in feature_columns(df.columns) if not pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Target leakage: columnas POST
# ---------------------------------------------------------------------------

def test_patrones_post_reconocen_exactamente_las_columnas_post_de_runs(raw_runs):
    assert sorted(forbidden_columns(raw_runs.columns)) == sorted(POST_RUNS)


def test_patrones_post_reconocen_exactamente_las_columnas_post_de_races(raw_races):
    assert sorted(forbidden_columns(raw_races.columns)) == sorted(POST_RACES)


def test_salida_sin_columnas_post(v0):
    assert forbidden_columns(v0.columns) == []


def test_toda_columna_cruda_esta_justificada(v0, raw_runs, raw_races):
    """Cada columna cruda o se conserva, o se borra por POST. Ninguna
    desaparece sin razón documentada y no aparecen columnas sin explicar."""
    raw = set(raw_runs.columns) | set(raw_races.columns)
    dropped = raw - set(v0.columns)
    assert dropped == set(POST_RUNS) | set(POST_RACES)
    assert set(v0.columns) - raw == DERIVED


def test_features_de_v0_son_las_revisadas(v0):
    assert feature_columns(v0.columns) == EXPECTED_FEATURES_V0


def test_targets_resultado_y_mercado_no_son_features(v0):
    feats = set(feature_columns(v0.columns))
    for col in ("result", "won", "y_rel", "win_odds", "place_odds", "split"):
        assert col in v0.columns and col not in feats
    assert set(NON_FEATURES) <= set(v0.columns)


def test_toda_columna_tiene_descripcion_en_el_diccionario(v0):
    """Una columna nueva (por ejemplo, de v1) debe documentarse en
    data_dictionary.DESCRIPCIONES antes de entrar a la tabla."""
    assert [c for c in v0.columns if c not in DESCRIPCIONES] == []


def test_el_mercado_supera_los_umbrales(v0):
    """Control positivo: si win_odds no los superara, los umbrales no detectarían nada."""
    assert abs(_rho_en_carrera(v0, "win_odds")) > MAX_RHO
    assert abs(_auc_en_val(v0, "win_odds") - 0.5) > MAX_AUC_GAP


def test_ninguna_numerica_ordena_la_carrera_como_el_resultado(v0):
    rhos = {c: _rho_en_carrera(v0, c) for c in _numericas(v0)}
    assert {c: r for c, r in rhos.items() if abs(r) >= MAX_RHO} == {}


def test_ninguna_categorica_predice_al_ganador_como_el_mercado(v0):
    aucs = {c: _auc_en_val(v0, c) for c in _categoricas(v0)}
    assert {c: a for c, a in aucs.items() if abs(a - 0.5) >= MAX_AUC_GAP} == {}


def test_horse_rating_es_el_rating_previo_a_la_carrera(v0):
    """Ganar no debe asociarse al cambio de rating que POST (llega) a la carrera (si no, el rating
    ya incluiría el resultado), pero sí al que PRE (sale) de ella (el rating se actualiza después)."""
    d = v0.sort_values(["horse_id", "date", "race_id"])
    g = d.groupby("horse_id")["horse_rating"]
    won = d["won"].astype(float)
    llega = (d["horse_rating"] - g.shift(1)).astype(float).corr(won)
    sale = (g.shift(-1) - d["horse_rating"]).astype(float).corr(won)
    assert abs(llega) < 0.1
    assert sale > 0.2


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def test_won_recalculado_coincide_con_el_original(v0, raw_runs):
    orig = raw_runs.set_index(["race_id", "horse_no"])["won"]
    new = v0.set_index(["race_id", "horse_no"])["won"]
    assert (new.sort_index() == orig.sort_index()).all()


def test_targets_consistentes(v0):
    assert ((v0["y_rel"] == 3) == (v0["won"] == 1)).all()
    assert (v0.groupby("race_id")["won"].sum() >= 1).all()


# ---------------------------------------------------------------------------
# Fuga temporal: partición
# ---------------------------------------------------------------------------

def test_split_sigue_la_regla_por_anio(v0):
    year = v0["date"].dt.year
    assert (v0.loc[year <= TRAIN_LAST_YEAR, "split"] == "train").all()
    assert (v0.loc[year == VAL_YEAR, "split"] == "val").all()
    assert (v0.loc[year == TEST_YEAR, "split"] == "test").all()


def test_ninguna_carrera_cruza_splits(v0):
    assert (v0.groupby("race_id")["split"].nunique() == 1).all()


def test_splits_en_orden_cronologico_estricto(v0):
    first = v0.groupby("split")["date"].min()
    last = v0.groupby("split")["date"].max()
    assert last["train"] < first["val"]
    assert last["val"] < first["test"]


def test_tabla_ordenada_y_carreras_contiguas(v0):
    assert v0["date"].is_monotonic_increasing
    rid = v0["race_id"].to_numpy()
    assert 1 + int((rid[1:] != rid[:-1]).sum()) == v0["race_id"].nunique()


def test_orden_de_filas_no_depende_del_resultado(v0):
    """Dentro de la carrera las filas van por horse_no; si fueran por result, la fila delataría al ganador."""
    assert (v0.groupby("race_id")["horse_no"].diff().dropna() > 0).all()


def test_columnas_de_carrera_constantes_por_carrera(v0):
    for col in RACE_LEVEL:
        assert (v0.groupby("race_id")[col].nunique(dropna=False) == 1).all(), col


# ---------------------------------------------------------------------------
# Fuga por preprocesamiento: nada se ajusta con la tabla completa
# ---------------------------------------------------------------------------

def test_no_se_pierden_corredores(v0, raw_runs):
    assert len(v0) == len(raw_runs)


def test_valores_iguales_a_los_originales(v0, raw_runs, raw_races):
    """No se imputa, no se escala ni se codifica nada antes de particionar.
    Cada valor de v0 es el valor original, excepto las correcciones a NaN.
    Así ninguna estadística de val/test entra a la tabla."""
    raw = raw_runs.merge(raw_races, on="race_id", how="inner")
    raw["date"] = pd.to_datetime(raw["date"])
    key = ["race_id", "horse_no"]
    a = v0.set_index(key).sort_index()
    b = raw.set_index(key).sort_index()

    corrected = {
        "declared_weight": b["declared_weight"] < MIN_DECLARED_WEIGHT,
        "draw": b["draw"] > MAX_DRAW,
    }
    for col in sorted((set(a.columns) & set(b.columns)) - {"won"}):
        x, y = a[col], b[col]
        if col in corrected:
            assert x[corrected[col]].isna().all(), col
            x, y = x[~corrected[col]], y[~corrected[col]]
        assert (x.isna() == y.isna()).all(), col
        x, y = x[x.notna()], y[y.notna()]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            assert np.array_equal(x.astype(float).to_numpy(), y.astype(float).to_numpy()), col
        else:
            assert (x.astype(str) == y.astype(str)).all(), col


def test_correcciones_dejan_valores_en_rango(v0):
    assert v0["declared_weight"].min() >= MIN_DECLARED_WEIGHT
    assert v0["draw"].max() <= MAX_DRAW
