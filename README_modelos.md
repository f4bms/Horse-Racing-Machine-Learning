# Parte B — Modelos, métricas y calibración

## Qué hay aquí

| Archivo | Qué hace |
|---|---|
| `src/hkr/data.py` | Carga la tabla de A y valida el contrato (splits, carreras contiguas, columnas prohibidas) |
| `src/hkr/metrics.py` | Top-1, MRR, NDCG@k, AUC intra-carrera, log loss, Brier, ECE, tabla de confiabilidad |
| `src/hkr/models.py` | La escalera: random, market, logreg, gbm_binary, clogit, lambdamart |
| `src/hkr/calibration.py` | Softmax por carrera + temperatura ajustada en val |
| `src/hkr/tuning.py` | Optuna con forward chaining dentro de train |
| `src/hkr/analysis.py` | Curvas de aprendizaje y análisis de errores |
| `scripts/run_experiments.py` | Corre todo (`--stage dev` y luego `--stage final`) |
| `scripts/make_synthetic.py` | Datos falsos con el formato del contrato, solo para probar |
| `tests/test_core.py` | Tests de métricas, gradiente del clogit, grupos y contrato |

## Contrato con Persona A

A entrega `data/processed/dataset_vX.parquet` con **una fila por caballo-carrera** y:

| Columna | Tipo | Nota |
|---|---|---|
| `race_id` | int | |
| `date` | fecha | |
| `split` | `train` / `val` / `test` | ≤2003 / 2004 / 2005 |
| `won` | 0/1 | dead heats: dos 1 en la misma carrera |
| `y_rel` | 3/2/1/0 | ecuación 5 del paper |
| `win_odds` | float | solo para el baseline de mercado, nunca feature |
| resto | features | numéricas o texto; los NaN se dejan como NaN |

Cualquier columna que no esté en `RESERVED` (data.py) se toma como feature. Si
aparece algo como `finish_time`, `position_sec1` o `*_dividend`, el script se
detiene con error. Para fijar la lista a mano, poner `"features": [...]` en
`configs/experiment.json`.

## Cómo correr

```bash
pip install -r requirements.txt
python -m pytest -q tests

# desarrollo: tunea, entrena con train, calibra y reporta en val (repetible)
python scripts/run_experiments.py --data data/processed/dataset_v1.parquet --stage dev

# final: evalúa test UNA vez y deja results/TEST_EVALUATED.lock
python scripts/run_experiments.py --data data/processed/dataset_v1.parquet --stage final
```

Para probar sin la tabla real: `python scripts/make_synthetic.py` y usar
`data/synthetic/dataset_synth.parquet`. **Esos números no van al reporte.**

## Decisiones (para la defensa)

**Métricas por carrera, no por fila.** El campo va de 4 a 14; promediar por
fila pesa más las carreras grandes, y la pregunta es por ganador de carrera.
Accuracy por fila está descartada: no predecir nunca un ganador da 92 %.

**MRR como objetivo del tuning**, no top-1: top-1 es 0/1 por carrera y es
ruidoso; MRR también premia dejar al ganador segundo o tercero.

**Forward chaining dentro de train** (valida 2001, 2002, 2003). K-fold barajado
dejaría que carreras futuras informen al pasado. Val (2004) no se usa para
tunear, así queda limpia para comparar modelos y calibrar.

**Todos los baselines con hiperparámetros se tunean igual que LambdaMART.**
Tunear solo el modelo propuesto inflaría su ventaja.

**El número de árboles sale de early stopping en los folds**, y el modelo
final se reentrena con ese número fijo. Así val no se usa para early stopping.

**Calibración por temperatura en val.** Un solo parámetro, no cambia el orden
(top-1, MRR, NDCG idénticos), solo log loss, Brier y ECE. LambdaMART no da
probabilidades; sin este paso su log loss no significa nada.

**Logit condicional sin intercepto**: cualquier constante se cancela dentro de
la carrera. Es el escalón que aísla "las probabilidades suman 1".

**Sin class weighting ni SMOTE**: el desbalance es estructural (un ganador por
carrera); la normalización por carrera lo absorbe.

**Random reporta el top-1 esperado exacto** (promedio de 1/n), no una tirada.
Su probabilidad es exactamente 1/n.

**Intervalos de confianza por bootstrap de carreras.** Con ~770 carreras por
año, el error estándar de top-1 es ~1.5 puntos. Cada fila trae `top1_lo/hi`
(IC 95 %) y `top1_diff_vs_ref` con su IC: la diferencia pareada contra el
mercado, en las mismas carreras. Si el IC de la diferencia incluye 0, no se
puede afirmar que el modelo le gana al mercado.

## Relación con lo visto en clase

| En clase | Aquí |
|---|---|
| `Pipeline` + `ColumnTransformer` con `OneHotEncoder` antes del modelo | `preprocess.py`: igual, ajustado solo con train |
| `cross_validate` reportando media ± desviación estándar | `tuning.py` guarda MRR y top-1 con su desviación en cada trial |
| `GridSearchCV` para hiperparámetros | Optuna (permitido por el enunciado): explora rangos continuos con menos corridas que una grilla; todos los trials quedan en CSV |
| Árbol → bagging → random forest → boosting | La escalera usa boosting (LightGBM), la familia que el profe señaló como la más fuerte en datos tabulares |
| Boosting = modelos en serie que corrigen los errores del anterior | LambdaMART es boosting donde el "error" a corregir sale de los lambdas (cuánto cambia el NDCG si se intercambian dos caballos) |
| XGBoost: boosting con regularización | Se usa LightGBM, misma idea; se eligió porque trae `lambdarank`, maneja NaN y categóricas solo, y es rápido. `reg_lambda` es la regularización L2 y se tunea |
| `class_weight="balanced"` para desbalance | Descartado a propósito: el desbalance es estructural y se resuelve normalizando por carrera |
| "Siempre hay que hacer múltiples experimentos y comprobar de manera científica" | Escalera de 6 modelos + IC por bootstrap |
| Referencia: *An Introduction to Statistical Learning*, cap. 8 | Citable para árboles, bagging, random forest y boosting |

## Lo que sale en `results/`

- `val_metrics.csv`, `test_metrics.csv`: la tabla comparativa
- `tuning/*_trials.csv`, `tuning/*_best.json`: espacio, todos los trials, tiempo
- `learning_curve_*.csv`: train vs val al variar años de historia
- `val_errors_*`: top-1 por tamaño de campo, por cuota del ganador, y según coincida o no con el favorito
- `val_confident_misses_*`, `val_missed_winners_*`: casos límite para discutir
- `val_reliability_*`, `test_reliability_*`: datos del reliability diagram
- `literature_reference.csv`: Silverman y Chavan, con la nota de comparabilidad
