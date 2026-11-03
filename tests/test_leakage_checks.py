"""
Verificación "solo pasado" para las features históricas de dataset_v1 (punto 7 del TODO).

dataset_v0 NO tiene columnas de historial: no hay ninguna tasa de victoria ni
"días desde la última carrera". Esas features llegan con dataset_v1 (tasas de
victoria del caballo, jinete y entrenador; días desde la última carrera; indicador
"sin historial"). Este archivo deja lista desde ahora la prueba que van a tener
que pasar, para que v1 se construya contra una regla ya verificada.

Regla (TODO, "REGLAS"): una feature histórica solo puede usar carreras con fecha
ANTERIOR a la de la carrera que se predice. Las carreras del mismo día tampoco
cuentan, aunque se hayan corrido antes en la jornada.

Prueba de perturbación (assert_uses_only_past): para cada fecha de corte se
construyen las features dos veces, una con los resultados reales y otra con los
resultados de las carreras con fecha >= corte alterados. Si cambia alguna
feature de una fila con fecha <= corte, esa feature leyó resultados del mismo día
o del futuro. La prueba no depende de cómo esté implementada la feature (shift,
expanding, merge, target encoding...).

Los tests usan un dataset de juguete con un ejemplo correcto y tres fugas
típicas, para demostrar que la prueba acepta lo correcto y detecta lo incorrecto.
Cuando exista dataset_v1, aquí se agrega el test que aplica assert_uses_only_past
a la función que construye sus features.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pytest

TARGET_COLS = ("result", "won", "y_rel")


# ---------------------------------------------------------------------------
# Verificación reutilizable
# ---------------------------------------------------------------------------

def perturb_results(df: pd.DataFrame, cutoff, seed: int = 0) -> pd.DataFrame:
    """Copia de `df` con los resultados de las filas con fecha >= cutoff alterados:
    `won` se invierte y `result`/`y_rel` se permutan al azar."""
    out = df.copy()
    future = out["date"] >= pd.Timestamp(cutoff)
    rng = np.random.default_rng(seed)
    for col in TARGET_COLS:
        if col not in out.columns:
            continue
        if col == "won":
            out.loc[future, col] = 1 - out.loc[future, col]
        else:
            out.loc[future, col] = rng.permutation(out.loc[future, col].to_numpy())
    return out


def assert_uses_only_past(df: pd.DataFrame,
                          build_features: Callable[[pd.DataFrame], pd.DataFrame],
                          cutoffs: Iterable | None = None, seed: int = 0) -> None:
    """Falla (AssertionError) si, para algún corte, una feature de una fila con
    fecha <= corte depende de resultados de carreras con fecha >= corte.
    Por defecto usa como corte cada fecha de `df`."""
    if cutoffs is None:
        cutoffs = sorted(df["date"].unique())
    real = build_features(df)
    for cutoff in cutoffs:
        altered = build_features(perturb_results(df, cutoff, seed))
        past = df["date"] <= pd.Timestamp(cutoff)
        pd.testing.assert_frame_equal(real.loc[past], altered.loc[past], check_dtype=False,
                                      obj=f"features con corte {pd.Timestamp(cutoff).date()}")


# ---------------------------------------------------------------------------
# Dataset mock y features de ejemplo
# ---------------------------------------------------------------------------
# No se busca un dataframe realista, sino suficiente variación para probar las funciones

@pytest.fixture
def ds() -> pd.DataFrame: # template preconfigurada de un dataset
    rows = []
    for day in range(1, 5): # Del 1 al 4 de enero
        date = pd.Timestamp(f"2003-01-0{day}")
        for race in range(2): # 2 carreras por dia
            race_id = 2 * day + race
            for pos in range(3): # 3 caballos por carrera = 6; los mismos cada jornada
                horse = 3 * race + pos
                winner = (day + race + pos) % 3 == 0 # pseudo-aleatorio de win, sin prob de empate
                rows.append({"race_id": race_id, "date": date, "horse_id": horse,
                             "jockey_id": 10 + pos, "won": int(winner)}) # hay 3 jinetes en total
    return pd.DataFrame(rows).sort_values(["date", "race_id"]).reset_index(drop=True)


def _tasa_victoria_previa(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """CORRECTA (ejemplo de lo que calculará v1): tasa de victoria de `key` en
    jornadas estrictamente anteriores. Se agrega por (key, fecha) y se desplaza una
    jornada, así las carreras del mismo día no cuentan."""
    daily = df.groupby([key, "date"])["won"].agg(["sum", "count"]).groupby(level=0).cumsum()
    prev = daily.groupby(level=0).shift(1)
    rate = (prev["sum"] / prev["count"]).rename(f"{key}_win_rate").reset_index()
    out = df[[key, "date"]].merge(rate, on=[key, "date"], how="left")
    out.index = df.index # recupera el indice original
    return out[[f"{key}_win_rate"]] # solo se devuelve la feature calculada


def _fuga_incluye_carrera_actual(df: pd.DataFrame) -> pd.DataFrame:
    """FUGA: el promedio expansivo incluye la propia carrera (falta el shift)."""
    rate = df.groupby("horse_id")["won"].transform(lambda s: s.expanding().mean())
    return rate.to_frame("horse_win_rate")


def _fuga_mismo_dia(df: pd.DataFrame) -> pd.DataFrame:
    """FUGA: shift(1) por fila usa la carrera anterior del jinete en el mismo día."""
    rate = df.groupby("jockey_id")["won"].transform(lambda s: s.shift(1).expanding().mean())
    return rate.to_frame("jockey_win_rate")


def _fuga_historial_completo(df: pd.DataFrame) -> pd.DataFrame:
    """FUGA: promedio con TODAS las carreras del jinete, incluidas las futuras.
    Es el error típico de un target encoding ajustado con toda la tabla (punto 8)."""
    rate = df.groupby("jockey_id")["won"].transform("mean")
    return rate.to_frame("jockey_win_rate")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_acepta_tasa_previa_por_caballo(ds):
    assert_uses_only_past(ds, lambda d: _tasa_victoria_previa(d, "horse_id"))


def test_acepta_tasa_previa_por_jinete_sin_contar_el_mismo_dia(ds):
    assert_uses_only_past(ds, lambda d: _tasa_victoria_previa(d, "jockey_id"))


@pytest.mark.parametrize("feature", [
    _fuga_incluye_carrera_actual, _fuga_mismo_dia, _fuga_historial_completo,
], ids=["incluye_carrera_actual", "usa_carreras_del_mismo_dia", "usa_historial_completo"])
def test_detecta_fugas_tipicas(ds, feature):
    with pytest.raises(AssertionError):
        assert_uses_only_past(ds, feature)
