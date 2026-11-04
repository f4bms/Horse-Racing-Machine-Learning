"""
Genera data/processed/dataset_v0.parquet (o dataset_v1) y .csv a partir de races.csv y runs.csv.
Uso:
    python src/build_dataset.py --raw-dir data/raw --out data/processed/dataset_v0.parquet
    python src/build_dataset.py --raw-dir data/raw --out data/processed/dataset_v0.parquet --csv
    python src/build_dataset.py --version v1 --csv   # con historial; guarda también docs/seleccion_k_dataset_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cleaning import build_v0
from history import build_v1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--version", choices=["v0", "v1"], default="v0", help="v1 agrega el historial")
    ap.add_argument("--out", default=None, help="por defecto data/processed/dataset_<version>.parquet")
    ap.add_argument("--csv", action="store_true", help="guardar también una copia .csv")
    args = ap.parse_args()

    if args.version == "v0":
        df, report = build_v0(args.raw_dir)
    else:
        df, report, fit = build_v1(args.raw_dir)

    out = Path(args.out or f"data/processed/dataset_{args.version}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False) # sin guardar el index propio dle df
    if args.csv:
        df.to_csv(out.with_suffix(".csv"), index=False)

    print(f"filas: {len(df)}  carreras: {df['race_id'].nunique()}  "
          f"fechas: {df['date'].min().date()} -> {df['date'].max().date()}")
    by_split = df.groupby("split").agg(filas=("race_id", "size"), carreras=("race_id", "nunique"),
                                       desde=("date", "min"), hasta=("date", "max"))
    print(by_split.loc[["train", "val", "test"]].to_string())
    print("carreras con dead heat:", int((df.groupby("race_id")["won"].sum() > 1).sum())) # mas de un top 1
    print("Correcciones a NaN:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if args.version == "v1":
        print(f"historial: prior (tasa de victoria en train) = {fit['prior']:.4f}, k elegido en val = {fit['k']}")
        search = Path("docs") / "seleccion_k_dataset_v1.csv"
        search.parent.mkdir(exist_ok=True)
        fit["search"].to_csv(search, index=False)
        print(f"búsqueda de k guardada en {search}")
    print(f"columnas ({df.shape[1]}): {list(df.columns)}")
    print(f"guardado en {out}")


if __name__ == "__main__":
    main()
