"""Antes y después del entrenamiento, para los 4 modelos que aprenden.

Usa los modelos ya entrenados (results/models.joblib) y el año de validación
(2004), que ningún modelo vio al entrenar. Para cada modelo mide cuántos
ganadores acierta:

  ANTES   = el mismo modelo sin lo aprendido
            - logreg y clogit: todos los pesos en 0
            - gbm_binary y lambdamart: 0 árboles (mismo puntaje para todos)
            En todos los casos acierta lo mismo que el azar.
  DESPUÉS = el modelo con lo que aprendió en 1997-2003.

Para los modelos de árboles también muestra el progreso a medida que se
agregan árboles: es el entrenamiento visto paso a paso.

Uso (después de run_experiments.py --stage dev):
  python scripts/antes_despues.py
  python scripts/antes_despues.py --data data/processed/dataset_v1.parquet --models results/models.joblib
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from hkr import data as D  # noqa: E402
from hkr import metrics as M  # noqa: E402


def aciertos(val, score):
    return M.top1_accuracy(val, score) * val["race_id"].nunique()


def antes_lineal(name, model, val):
    """Pone los pesos en 0, mide, y los devuelve como estaban."""
    if name == "logreg":
        clf = model.pipe_.named_steps["clf"]
        w, b = clf.coef_.copy(), clf.intercept_.copy()
        clf.coef_[:] = 0
        clf.intercept_[:] = 0
        s = model.score(val)
        clf.coef_[:], clf.intercept_[:] = w, b
    else:  # clogit
        w = model.coef_.copy()
        model.coef_[:] = 0
        s = model.score(val)
        model.coef_[:] = w
    return s


def scores_arbol(name, model, val, k):
    """Puntaje usando solo los primeros k árboles (k=0: ninguno)."""
    if k == 0:
        return np.zeros(len(val))
    X = model.enc_.transform(val[model.features])
    if name == "gbm_binary":
        return model.clf_.predict_proba(X, num_iteration=k)[:, 1]
    return model.model_.predict(X, num_iteration=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset_v1.parquet")
    ap.add_argument("--models", default="results/models.joblib")
    a = ap.parse_args()

    val = D.split(D.load(a.data), "val")
    n = val["race_id"].nunique()
    arts = joblib.load(a.models)
    azar = M.top1_accuracy(val, np.zeros(len(val))) * n

    print(f"\nExamen: {n} carreras de 2004 que ningún modelo vio al entrenar")
    print(f"Escogiendo al azar se aciertan, en promedio, {azar:.0f}\n")
    print(f"{'modelo':<12}{'ANTES de entrenar':>20}{'DESPUÉS de entrenar':>22}{'ganancia':>11}")
    print("-" * 65)

    progreso = {}
    for name in ["logreg", "clogit", "gbm_binary", "lambdamart"]:
        if name not in arts:
            continue
        m = arts[name]["model"]
        if name in ("logreg", "clogit"):
            antes = aciertos(val, antes_lineal(name, m, val))
            despues = aciertos(val, m.score(val))
        else:
            total = m.model_.booster_.num_trees() if name == "lambdamart" else m.clf_.booster_.num_trees()
            pasos = sorted({0, 1, 2, 5, 10, 25, 50, 100, total} & set(range(total + 1)))
            progreso[name] = [(k, aciertos(val, scores_arbol(name, m, val, k))) for k in pasos]
            antes, despues = progreso[name][0][1], progreso[name][-1][1]
        print(f"{name:<12}{antes:>13.0f} ({100*antes/n:4.1f} %){despues:>14.0f} ({100*despues/n:4.1f} %)"
              f"{despues - antes:>+9.0f}")

    for name, pts in progreso.items():
        print(f"\n{name}: aciertos a medida que se agregan árboles")
        for k, h in pts:
            barra = "#" * int(round(h / 5))
            print(f"  {k:>4} árboles: {h:>4.0f}  {barra}")
    print()


if __name__ == "__main__":
    main()
