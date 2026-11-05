"""
Genera el diccionario de datos de una tabla procesada (dataset_v0, v1).

Todo lo que es medible sale a partir de los datos: tipos de columnas, valores faltantes,
cantidad de valores únicos y rangos numéricos o categorías.

El rol de cada columna se determina mediante cleaning.NON_FEATURES, y de
DESCRIPCIONES se sacan las descripciones, de acuerdo a lo investigado de
los datasets

Uso:
    python src/data_dictionary.py
    python src/data_dictionary.py --data data/processed/dataset_v0.parquet --out-dir docs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cleaning import MAX_DRAW, MIN_DECLARED_WEIGHT, forbidden_columns

SPLITS = ["train", "val", "test"]

# Con hasta este número de valores distintos se listan todos con su conteo.
MAX_LISTED = 12
TOP_CATEGORIES = 5

# Columnas de la salida que NO son features, con su rol, todo lo demás es feature
NON_FEATURES = {
    "race_id": "id", "date": "id", "split": "partición",
    "won": "target", "y_rel": "target", "win_odds": "mercado",
    "horse_id": "id", "horse_no": "id", "race_no": "id",
    "result": "resultado (solo targets)", "place_odds": "mercado",
}

# Texto para columnas sin entrada en DESCRIPCIONES (por ejemplo, features nuevas de v1).
SIN_DESCRIPCION = "sin descripción"


def feature_columns(columns) -> list[str]:
    """Columnas que entran al modelo: todas menos las de NON_FEATURES."""
    return [c for c in columns if c not in NON_FEATURES]

DESCRIPCIONES = {
    "race_id": "Identificador de la carrera.",
    "date": "Fecha de la carrera.",
    "split": "Partición cronológica: train (año ≤ 2003), val (2004), test (2005).",
    "won": "1 si llegó 1.º (en empate por el primer lugar hay dos 1). Se recalcula desde `result`",
    "y_rel": "Relevancia: 3 ganó, 2 llegó 2.º–3.º, 1 llegó 4.º–6.º, 0 el resto.",
    "win_odds": "Cuota final. Solo para el baseline de mercado, no se usa como feature.",
    "horse_id": "Identificador del caballo.",
    "horse_no": "El número asignado al caballo en la carrera.",
    "race_no": "Número de la carrera, en el daí del encuentro pueden haber msá de una carrera y así se identifican.",
    "result": "Posición de llegada (POST). Solo se usa para construir `won` y `y_rel`.",
    "place_odds": "Probabilidad del caballo de llegar 1º-3º antes de iniciar la carrera.",
    "horse_age": "Edad del caballo.",
    "horse_country": "País o región de origen del caballo.",
    "horse_type": "Sexo del caballo, ej: *Gelding*, *Mare*, *Horse*, *Rig*, *Colt*, *Filly*",
    "horse_rating": "Calificación durante la carrera asignada a cada cabllo por la HKJC.",
    "horse_gear": "Equipo del caballo; `--` significa sin equipo.",
    "declared_weight": f"Peso declarado (lbs, con el equipo del jockey). Los valores < {MIN_DECLARED_WEIGHT} se anulan a NaN (error de registro).",
    "actual_weight": "Peso que lleva el caballo (en lbs).",
    "draw": f"Cajón de salida. Los valores > {MAX_DRAW} se anulan a NaN (no completaron el registro o no se presentaron).",
    "trainer_id": "Identificador del entrenador en el momento de la carrera, guardado como texto (categoría).",
    "jockey_id": "Identificador del jinete en dicha carrera, guardado como texto (categoría).",
    "venue": "Cadena de 2 carácteres que indica en cuál de los 2 circuitos se dio la carrera; ST = Shatin, HV = Happy Valley.",
    "config": "Configuración de la pista, relacionada principalmente con la posición del carril interior.",
    "surface": "Número que representa el tipo de superficie del circuito; 1=tierra, 0=césped.",
    "distance": "Distancia de la carrera.",
    "going": "Estado de la pista.",
    "horse_ratings": "Rango de calificación de los caballos que van a participar; *G* marca carreras especiales/de grupo.",
    "prize": "Premio de la carrera. Falta en un bloque de carreras de 1997-07-20 a 1997-10-15 y de 1997-12-13 a 1998-03-18.",
    "race_class": "Código de clase; 0-6, y 11–13 son carreras especiales/de grupo, no es una escala ordenada.",
    "field_size": "Corredores en la carrera (derivada de filas por race_id).",
    # Historial (dataset_v1, src/history.py): solo usan carreras de fechas anteriores.
    "horse_win_rate": "Tasa de victoria previa del caballo, con shrinkage hacia la tasa de train (k elegido en val).",
    "jockey_win_rate": "Tasa de victoria previa del jinete, con shrinkage hacia la tasa de train (k elegido en val).",
    "trainer_win_rate": "Tasa de victoria previa del entrenador, con shrinkage hacia la tasa de train (k elegido en val).",
    "horse_days_since_last": "Días desde la carrera anterior del caballo; NaN en su primera carrera del dataset.",
    "horse_no_history": "1 si es la primera carrera del caballo en el dataset (sin historial previo).",
    "gear_any": "1 si el caballo lleva algún equipo (horse_gear distinto de `--`). Reemplaza a horse_gear en v1.",
}


def _fmt(x) -> str:
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    if isinstance(x, float) and x.is_integer(): # comprueba que sea decimal, y luego que no sea flotante
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _values(s: pd.Series) -> str:
    s = s.dropna()
    counts = s.value_counts()
    numeric = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
    if len(counts) <= MAX_LISTED:
        if numeric:
            counts = counts.sort_index()
        return ", ".join(f"{_fmt(v)} ({n})" for v, n in counts.items())
    if numeric:
        return f"{_fmt(s.min())} – {_fmt(s.max())}"
    top = ", ".join(f"{_fmt(v)} ({n})" for v, n in counts.head(TOP_CATEGORIES).items())
    return f"{len(counts)} categorías; top: {top}"


def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    post = set(forbidden_columns(df.columns))
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "columna": c,
            "rol": "PROHIBIDA (POST)" if c in post else NON_FEATURES.get(c, "feature"),
            "tipo": str(s.dtype),
            "nan": int(s.isna().sum()),
            "nan_pct": round(100 * s.isna().mean(), 2),
        }
        if "split" in df.columns:
            for sp in SPLITS:
                row[f"nan_{sp}_pct"] = round(100 * s[df["split"] == sp].isna().mean(), 2)
        row["unicos"] = int(s.nunique())
        row["valores"] = _values(s)
        row["descripcion"] = DESCRIPCIONES.get(c, SIN_DESCRIPCION)
        rows.append(row)
    return pd.DataFrame(rows)


def _cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def to_markdown(dic: pd.DataFrame, df: pd.DataFrame, source: Path) -> str:
    n_feat = int((dic["rol"] == "feature").sum())
    lines = [
        f"# Diccionario de datos: `{source.name}`",
        "",
        f"Generado por `src/data_dictionary.py` desde `{source.as_posix()}`. No editar a mano: regenerar después de reconstruir la tabla.",
        "",
        f"- Filas: {len(df)}",
        f"- Columnas: {df.shape[1]} ({n_feat} features)",
    ]
    if "race_id" in df.columns:
        lines.append(f"- Carreras: {df['race_id'].nunique()}")
    if "date" in df.columns:
        lines.append(f"- Fechas: {_fmt(df['date'].min())} → {_fmt(df['date'].max())}")
    if "split" in df.columns:
        by = df.groupby("split")
        for sp in SPLITS:
            if sp in by.groups:
                g = by.get_group(sp)
                lines.append(f"- {sp}: {len(g)} filas, {g['race_id'].nunique()} carreras, "
                             f"{_fmt(g['date'].min())} → {_fmt(g['date'].max())}")
    lines += [
        "",
        "Roles: `feature` entra al modelo; `id`, `partición`, `target`, `mercado` y `resultado` se conservan pero nunca son features.",
        "",
    ]

    split_cols = [f"nan_{sp}_pct" for sp in SPLITS if f"nan_{sp}_pct" in dic.columns]
    header = ["Columna", "Rol", "Tipo", "NaN", "NaN %"]
    if split_cols:
        header.append("NaN % train / val / test")
    header += ["Únicos", "Valores (conteo) o rango", "Descripción"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, r in dic.iterrows():
        cells = [f"`{r['columna']}`", r["rol"], r["tipo"], r["nan"], r["nan_pct"]]
        if split_cols:
            cells.append(" / ".join(str(r[c]) for c in split_cols))
        cells += [r["unicos"], r["valores"], r["descripcion"]]
        lines.append("| " + " | ".join(_cell(x) for x in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset_v0.parquet")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()

    source = Path(args.data)
    df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source, parse_dates=["date"])
    dic = build_dictionary(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"diccionario_{source.stem}"
    (out_dir / f"{stem}.md").write_text(to_markdown(dic, df, source), encoding="utf-8")
    dic.to_csv(out_dir / f"{stem}.csv", index=False)

    post = dic.loc[dic["rol"] == "PROHIBIDA (POST)", "columna"].tolist()
    if post:
        print(f"AVISO: columnas POST en la tabla: {post}")
    missing = dic.loc[dic["descripcion"] == SIN_DESCRIPCION, "columna"].tolist()
    if missing:
        print(f"AVISO: columnas sin descripción configurada: {missing}")
    print(f"columnas: {len(dic)}  features: {int((dic['rol'] == 'feature').sum())}")
    print(f"guardado en {out_dir / stem}.md y .csv")


if __name__ == "__main__":
    main()
