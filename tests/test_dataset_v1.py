"""
Tests de dataset_v1 (dataset_v0 + historial).

La regla "solo pasado" de las features históricas se prueba en test_leakage_checks.py.
Aquí se revisa que v1 no cambie nada de v0, que el ajuste del historial no mire test
y que las features nuevas sean coherentes y no sean un proxy del resultado.
"""
from __future__ import annotations

import pandas as pd

from data_dictionary import DESCRIPCIONES, feature_columns
from history import K_GRID, NEW_FEATURES, REPLACED, fit_history
# Mismos umbrales y medidas que los controles de proxy de v0.
from test_leakage_v0 import EXPECTED_FEATURES_V0, MAX_RHO, _rho_en_carrera


def test_v1_es_v0_mas_el_historial(v0, v1_build):
    v1, _ = v1_build
    base = [c for c in v0.columns if c not in REPLACED]
    # compara las columnas de v0 sin las reemplazadas, más las nuevas de v1
    assert list(v1.columns) == base + NEW_FEATURES
    # compara los df en las columnas comunes, por lo que deberian de ser identicos (mismo valores e index)
    pd.testing.assert_frame_equal(v1[base], v0[base])


def test_columnas_reemplazadas_no_estan_en_v1(v1_build):
    """El código de modelos toma como feature toda columna que no excluya; por eso
    jockey_id, trainer_id y horse_gear se quitan de la tabla y no solo se marcan."""
    v1, _ = v1_build
    assert not set(REPLACED) & set(v1.columns)


def test_columnas_reemplazadas_se_recuperan_de_v0_por_indice(v0, v1_build):
    """v0 y v1 tienen las mismas filas con el mismo índice, así que lo quitado se
    recupera linealmente. Se comprueba con gear_any."""
    v1, _ = v1_build
    gear = v0.loc[v1.index, "horse_gear"]
    assert (v1["gear_any"] == (gear != "--").astype(int)).all()


def test_features_de_v1_son_las_revisadas(v1_build):
    v1, _ = v1_build
    expected = [c for c in EXPECTED_FEATURES_V0 if c not in REPLACED] + NEW_FEATURES
    assert feature_columns(v1.columns) == expected


def test_toda_columna_de_v1_tiene_descripcion(v1_build):
    v1, _ = v1_build
    assert [c for c in v1.columns if c not in DESCRIPCIONES] == []


def test_ajuste_del_historial_no_mira_test(v0, v1_build):
    """Cambiar todos los resultados de test no cambia ni el prior ni los k."""
    _, fit = v1_build
    altered = v0.copy()
    test = altered["split"] == "test"
    altered.loc[test, "won"] = 1 - altered.loc[test, "won"]
    refit = fit_history(altered)
    assert refit["prior"] == fit["prior"]
    assert refit["k"] == fit["k"]


def test_k_elegido_esta_en_el_grid(v1_build):
    _, fit = v1_build
    assert set(fit["k"].values()) <= set(K_GRID) # <= verifica que sea subconjunto del grid


def test_sin_historial_es_la_primera_carrera_del_caballo(v1_build):
    v1, _ = v1_build
    first = v1.groupby("horse_id").cumcount() == 0
    assert (v1["horse_no_history"] == first.astype(int)).all()
    assert (v1["horse_days_since_last"].isna() == first).all()


def test_dias_desde_la_ultima_carrera_es_de_una_fecha_anterior(v1_build):
    v1, _ = v1_build
    assert (v1["horse_days_since_last"].dropna() >= 1).all()


def test_tasas_entre_0_y_1_y_sin_nan(v1_build):
    v1, _ = v1_build
    rates = v1[["horse_win_rate", "jockey_win_rate", "trainer_win_rate"]]
    assert rates.notna().all().all()
    assert ((rates >= 0) & (rates <= 1)).all().all()


def test_ninguna_feature_nueva_ordena_la_carrera_como_el_resultado(v1_build):
    v1, _ = v1_build
    rhos = {c: _rho_en_carrera(v1, c) for c in NEW_FEATURES}
    assert {c: r for c, r in rhos.items() if abs(r) >= MAX_RHO} == {}
