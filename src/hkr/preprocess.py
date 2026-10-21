"""Preprocesamiento dentro de pipelines que se ajustan SOLO con entrenamiento.

Dos variantes:
- Lineal (logística, logit condicional): imputación + escalado + one-hot.
- Árboles (LightGBM): solo fija las categorías vistas en train; LightGBM
  maneja NaN de forma nativa, así que no se imputa (un debutante no es un
  caballo promedio, Sección V-C del paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in features
           if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    num = [c for c in features if c not in cat]
    return num, cat


def linear_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop", verbose_feature_names_out=False,
    )


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa `col` con la mediana de su grupo (p. ej. prize dentro de race_class),
    aprendida SOLO en train, y agrega `<col>_unrecorded` = 1 donde faltaba.

    Sección V-C del paper: prize falta en un bloque contiguo de 1997-98, así que
    una mediana global metería niveles de premio de años posteriores; el premio
    depende sobre todo de la clase, por eso se imputa dentro de race_class.
    """

    def __init__(self, col: str = "prize", by: str = "race_class"):
        self.col = col
        self.by = by

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.groupby(self.by)[self.col].median().to_dict()
        self.global_ = float(X[self.col].median())
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns_].copy()
        miss = X[self.col].isna()
        X[f"{self.col}_unrecorded"] = miss.astype(int)
        fill = X[self.by].map(self.medians_).fillna(self.global_)
        X.loc[miss, self.col] = fill[miss]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_ + [f"{self.col}_unrecorded"], dtype=object)


def linear_pipeline_prep(train_df: pd.DataFrame, features: list[str]) -> Pipeline:
    """Preprocesamiento de los modelos lineales (logística y logit condicional)."""
    num, cat = split_types(train_df, features)
    steps = []
    if "prize" in features and "race_class" in features:
        steps.append(("prize_by_class", GroupMedianImputer("prize", "race_class")))
        num = num + ["prize_unrecorded"]
    steps.append(("ct", linear_preprocessor(num, cat)))
    return Pipeline(steps)


class TreeCategoricals(BaseEstimator, TransformerMixin):
    """Convierte categóricas a `category` con las categorías vistas en train.

    Una categoría que aparece por primera vez en val/test queda como NaN, que
    es lo correcto: el modelo no la conoce.
    """

    def __init__(self, cat: list[str] | None = None):
        self.cat = cat

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in (self.cat or [])}
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columns_].copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna(), np.nan), categories=cats)
        return X
