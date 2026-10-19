"""Genera data/processed/dataset_v0.parquet y .csv a partir de races.csv y runs.csv.
Uso:
    python scripts/build_dataset.py --raw-dir data/raw --out data/processed/dataset_v0.parquet
    python scripts/build_dataset.py --raw-dir data/raw --out data/processed/dataset_v0.parquet --csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cleaning import build_v0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/dataset_v0.parquet")
    ap.add_argument("--csv", action="store_true", help="guardar también una copia .csv")
    args = ap.parse_args()

    df, report = build_v0(args.raw_dir)

    out = Path(args.out)
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
    print("verification Warnings:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"columnas ({df.shape[1]}): {list(df.columns)}")
    print(f"guardado en {out}")


if __name__ == "__main__":
    main()
