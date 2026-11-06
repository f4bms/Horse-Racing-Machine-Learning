"""
Tests unitarios de cleaning.py con datos sintéticos (no necesitan data/raw).

Cada test comprueba que un safeguard contra fuga o error de datos se active
cuando debe: si alguien la desactiva o la rompe, el test falla.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cleaning import (
    MAX_DRAW, MIN_DECLARED_WEIGHT, DataError, add_split, add_targets,
    drop_post, forbidden_columns, merge, prepare, validate,
)
# Los roles de las columnas estan en data_dictionary.py
from data_dictionary import NON_FEATURES, feature_columns


# ---------------------------------------------------------------------------
# Reconocimiento y borrado de columnas POST
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", [
    "finish_time", "lengths_behind", "position_sec1", "behind_sec6", "time1",
    "sec_time7", "win_dividend1", "place_dividend4", "win_combination2", "place_combination1",
])
def test_forbidden_columns_detecta_post(col):
    assert forbidden_columns([col]) == [col]


@pytest.mark.parametrize("col", [
    "date", "draw", "horse_ratings", "horse_rating", "race_class", "prize",
    "win_odds", "place_odds", "distance", "field_size", "result",
])
def test_forbidden_columns_no_marca_columnas_pre(col):
    # result no es PRE, pero se conserva a propósito para los targets
    # control es que no se eliminan columnas requeridas
    assert forbidden_columns([col]) == []


def test_drop_post_borra_todas_las_post():
    df = pd.DataFrame(columns=["race_id", "time1", "win_dividend1", "draw"])
    assert list(drop_post(df, "x").columns) == ["race_id", "draw"]


def test_merge_rechaza_post_antes_del_join():
    runs = pd.DataFrame({"race_id": [1], "horse_no": [1]})
    races = pd.DataFrame({"race_id": [1], "win_combination1": [1]})
    with pytest.raises(DataError):
        merge(runs, races)


def test_merge_rechaza_si_el_join_pierde_corredores():
    runs = pd.DataFrame({"race_id": [1, 2], "horse_no": [1, 1]})
    races = pd.DataFrame({"race_id": [1], "venue": ["ST"]})
    with pytest.raises(DataError):
        merge(runs, races)


# ---------------------------------------------------------------------------
# Roles: targets, resultado y mercado nunca son features
# ---------------------------------------------------------------------------

def test_non_features_incluye_targets_resultado_y_mercado():
    for col in ("won", "y_rel", "result", "win_odds", "place_odds", "split", "race_id", "date"):
        assert col in NON_FEATURES


def test_feature_columns_excluye_non_features():
    cols = list(NON_FEATURES) + ["draw", "distance"]
    assert feature_columns(cols) == ["draw", "distance"]


# ---------------------------------------------------------------------------
# Correcciones de prepare(): el valor imposible pasa a NaN y la fila se conserva
# ---------------------------------------------------------------------------

# **over permite recibir una cantidad variable de argumentos nombrados
def _runs_for_prepare(**over) -> pd.DataFrame:
    df = pd.DataFrame({
        "race_id": [1, 1, 1], "horse_no": [1, 2, 3],
        "declared_weight": [1100.0, 1050.0, 1000.0], "draw": [1, 2, 3],
        "jockey_id": [10, 11, 12], "trainer_id": [20, 21, 22],
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_prepare_anula_peso_imposible_sin_borrar_la_fila():
    df = _runs_for_prepare(declared_weight=[MIN_DECLARED_WEIGHT - 1.0, 1050.0, MIN_DECLARED_WEIGHT])
    out, report = prepare(df)
    assert len(out) == 3
    assert out["declared_weight"].isna().tolist() == [True, False, False]
    assert report[f"declared_weight < {MIN_DECLARED_WEIGHT}"] == 1


def test_prepare_anula_draw_imposible_y_mantiene_entero():
    df = _runs_for_prepare(draw=[MAX_DRAW + 1, MAX_DRAW, 1])
    out, _ = prepare(df)
    assert out["draw"].isna().tolist() == [True, False, False]
    assert str(out["draw"].dtype) == "Int64"


def test_prepare_field_size_cuenta_las_filas_de_la_carrera():
    out, _ = prepare(_runs_for_prepare())
    assert out["field_size"].tolist() == [3, 3, 3]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def test_add_targets_dead_heat_y_niveles_de_y_rel():
    df = pd.DataFrame({"result": [1, 1, 3, 4, 6, 7, 14]})
    out = add_targets(df)
    assert out["won"].tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert out["y_rel"].tolist() == [3, 3, 2, 1, 1, 0, 0]


# ---------------------------------------------------------------------------
# Partición cronológica
# ---------------------------------------------------------------------------

def test_add_split_por_anio_calendario():
    df = pd.DataFrame({"date": pd.to_datetime(["2003-12-31", "2004-01-01", "2004-12-31", "2005-01-01"])})
    assert add_split(df)["split"].tolist() == ["train", "val", "val", "test"]


def test_add_split_rechaza_anios_fuera_de_la_particion():
    df = pd.DataFrame({"date": pd.to_datetime(["2006-01-01"])})
    with pytest.raises(DataError):
        add_split(df)


# ---------------------------------------------------------------------------
# validate(): cada regla se dispara con un caso que la viola
# ---------------------------------------------------------------------------

def _valid_table() -> pd.DataFrame:
    """Tres carreras de dos corredores, una por split, ya ordenadas."""
    return pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "date": pd.to_datetime(["2003-05-01"] * 2 + ["2004-05-01"] * 2 + ["2005-05-01"] * 2),
        "split": ["train"] * 2 + ["val"] * 2 + ["test"] * 2,
        "won": [1, 0, 0, 1, 1, 0],
        "y_rel": [3, 2, 2, 3, 3, 2],
        "win_odds": [2.0, 3.0, 4.0, 1.5, 2.5, 6.0],
    })


def test_validate_acepta_tabla_valida():
    validate(_valid_table())


def test_validate_rechaza_columna_post():
    df = _valid_table().assign(finish_time=1.0)
    with pytest.raises(DataError, match="POST"):
        validate(df)


def test_validate_rechaza_columna_obligatoria_faltante():
    with pytest.raises(DataError, match="obligatorias"):
        validate(_valid_table().drop(columns=["y_rel"]))


# Los corredores de una misma carrera deben pertenecer a la misma partición
def test_validate_rechaza_carrera_partida_entre_splits():
    df = _valid_table()
    df.loc[1, "split"] = "val"
    with pytest.raises(DataError, match="cruzan"):
        validate(df)


def test_validate_rechaza_splits_fuera_de_orden_cronologico():
    df = _valid_table()
    df.loc[df["race_id"] == 2, "date"] = pd.Timestamp("2002-01-01")
    with pytest.raises(DataError, match="cronológico"):
        validate(df)


def test_validate_rechaza_carrera_sin_ganador():
    df = _valid_table()
    df.loc[df["race_id"] == 1, "won"] = 0
    with pytest.raises(DataError, match="sin ganador"):
        validate(df)


def test_validate_rechaza_filas_de_carrera_no_contiguas():
    # Filas de las carreras 1 y 2 intercaladas: 1, 2, 1, 2, 3, 3.
    df = _valid_table().iloc[[0, 2, 1, 3, 4, 5]].reset_index(drop=True)
    with pytest.raises(DataError, match="contiguas"):
        validate(df)
