"""
Fixtures compartidas.

La tabla v0 se construye una sola vez por sesión desde data/raw, con el mismo
código que usa build_dataset.py, para que los tests verifiquen lo que realmente
se entrega y no una copia guardada en disco. Si faltan los CSV crudos, los
tests que los necesitan se saltan (los unitarios con datos sintéticos corren
igual).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cleaning import build_v0
from history import build_v1

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _require_raw() -> None:
    for name in ("races.csv", "runs.csv"):
        if not (RAW_DIR / name).exists():
            pytest.skip(f"falta {RAW_DIR / name}") # Interrumpe la prueba actual y la marca como omitida (skipped)


@pytest.fixture(scope="session")
def raw_runs() -> pd.DataFrame:
    _require_raw()
    return pd.read_csv(RAW_DIR / "runs.csv")


@pytest.fixture(scope="session")
def raw_races() -> pd.DataFrame:
    _require_raw()
    return pd.read_csv(RAW_DIR / "races.csv")


@pytest.fixture(scope="session")
def v0() -> pd.DataFrame:
    _require_raw()
    df, _ = build_v0(RAW_DIR)
    return df


@pytest.fixture(scope="session")
def v1_build():
    """(tabla v1, ajuste del historial: prior, k y search de k)."""
    _require_raw()
    df, _, fit = build_v1(RAW_DIR)
    return df, fit
