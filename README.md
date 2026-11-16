# Horse Racing & Machine Learning

Proyecto del curso de Inteligencia Artificial. Se eligió el Track A: Machine Learning clásico sobre datos tabulares. Se busca predecir el ganador de cada carrera de caballos en Hong Kong (HKJC,1997–2005) ordenando a los caballos de cada carrera, y compradando el modelo LambdaMART contra baselines, contra el favorito del mercado y contra la literatura.

- **Dataset:** dataset de Kaggle [`gdaley/hkracing`](https://www.kaggle.com/datasets/gdaley/hkracing/data): 79 447 filas (un caballo en una carrera) y 6 348 carreras.
- **Cuaderno principal:** el cuaderno [`notebooks/Etapa2_Analisis.ipynb`](notebooks/Etapa2_Analisis.ipynb), cuenta con el ciclo completo en tres partes: datos, modelos y resultados, y explicabilidad.
- **Declaración de uso de IA:** [`DECLARACION_IA.md`](DECLARACION_IA.md).

| Parte | Responsable | Código |
|---|---|---|
| Datos, limpieza, historial y pruebas de fuga | Roy Silva | `src/*.py`, `tests/test_cleaning_unit.py`, `tests/test_leakage_*.py`, `tests/test_dataset_v1.py` |
| Modelos, métricas, calibración y evaluación | Jorge Gutiérrez | `src/hkr/`, `scripts/`, `tests/test_core.py` |
| Explicabilidad (SHAP), documentación y analisis y revisión de resultados | Fabiola Meléndez | Parte 3 del cuaderno |

---

## Resultado principal

| Modelo | Acierta al ganador (top-1) | IC 95 % | Diferencia contra el mercado [IC 95 %] |
|---|---|---|---|
| Azar (1/n) | 8.1 % | 8.0–8.3 | −20.9 [−25.2, −16.8] |
| **Favorito del mercado** | **29.0 %** | 24.9–33.3 | — |
| Regresión logística | 22.1 % | 18.5–25.7 | −6.9 [−10.9, −3.1] |
| GBM binario | 19.1 % | 15.7–22.8 | −9.9 [−13.8, −5.8] |
| Logit condicional | 21.9 % | 18.3–25.7 | −7.1 [−11.1, −3.4] |
| LambdaMART (propuesto) | 21.5 % | 17.9–25.1 | −7.6 [−11.5, −3.6] |
| Silverman (2012), literatura | 21.6 % | — | no comparable directamente |

Los modelos que aprenden casi triplican al azar y alcanzan al baseline de literatura, peroninguno le gana al mercado, y el modelo simple de logística queda a la par de LambdaMART. El análisis completo se puede encontrar de manera más amplia en la Sección 2 del cuaderno.

---

## Estructura

```text
.
├── configs/experiment.json   # semilla, modelos, pruebas de Optuna y referencias de literatura
├── data/
│   ├── raw/                  # races.csv y runs.csv originales (Kaggle)
│   └── processed/            # salida: dataset_v0 y dataset_v1 (no se sube al repo)
├── docs/
│   ├── particion_y_fuga.md            # partición, variables descartadas por fuga y su verificación
│   ├── historial_v1.md                # features históricas de v1: definición, k y controles
│   ├── busqueda_hiperparametros.md    # espacio de búsqueda, protocolo, bordes y costo
│   ├── diccionario_dataset_v0.md/.csv # diccionarios generados (no editar a mano)
│   ├── diccionario_dataset_v1.md/.csv
│   ├── seleccion_k_dataset_v1.csv     # búsqueda de k del shrinkage (generada)
│   └── paper/                         # artículo (LaTeX) y metodología de modelado
├── instrucciones/            # enunciados del proyecto
├── notebooks/
│   └── Etapa2_Analisis.ipynb # cuaderno único de la Etapa 2
├── results/                  # salida de scripts/run_experiments.py con dataset_v1
├── results_v0/               # lo mismo con dataset_v0 (sin historial), para comparar
├── scripts/
│   ├── run_experiments.py    # tunea, entrena, calibra y evalúa (--stage dev / final)
│   ├── antes_despues.py      # aciertos de cada modelo antes y después de entrenar
│   └── make_synthetic.py     # datos falsos con el formato de la tabla, solo para probar
├── src/
│   ├── build_dataset.py      # genera data/processed/dataset_v0 o dataset_v1
│   ├── cleaning.py           # carga, borrado de columnas POST, preparación, targets y partición
│   ├── history.py            # features históricas de v1 (tasas con shrinkage, días, sin historial)
│   ├── data_dictionary.py    # genera el diccionario de datos en docs/
│   └── hkr/                  # modelos, métricas, calibración, tuning y análisis (ver abajo)
├── tests/                    # 89 pruebas: limpieza, fuga, historial, métricas y contrato
├── DECLARACION_IA.md
├── pytest.ini
└── requirements.txt
```

---

## Ejecución

**1. Entorno** (Python ≥ 3.10):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Datos crudos.** `data/raw/races.csv` y `data/raw/runs.csv`. Se pueden encontrar también en:  [Kaggle](https://www.kaggle.com/datasets/gdaley/hkracing/data).

**3. Pruebas** (no hace falta generar la tabla antes: la reconstruyen desde `data/raw`):

```bash
python -m pytest -q
```

**4. Generar las tablas:**

```bash
python src/build_dataset.py --version v0
python src/build_dataset.py --version v1
```

Opciones: `--raw-dir` (por defecto `data/raw`), `--out` (por defecto
`data/processed/dataset_<version>.parquet`), `--csv` (guarda también una copia `.csv`).
El script imprime filas, carreras y conteo por partición, y se detiene con `DataError` si
alguna validación falla. Con `v1` imprime además el prior y los *k* elegidos, y guarda la
búsqueda de *k* en `docs/seleccion_k_dataset_v1.csv`.

**5. Entrenamiento**:

```bash
python scripts/run_experiments.py --data data/processed/dataset_v1.parquet --stage dev
# opcional, para la comparación sin historial:
python scripts/run_experiments.py --data data/processed/dataset_v0.parquet --stage dev --out results_v0
```

Con validación cruzada dentro de train, reentrena con todo train, calibra en
validación (2004) y guarda todo en `results/`, incluido `results/models.joblib`. Al final imprime la tabla de validación (`logreg 0.2345`,
`market 0.3308`). LambdaMART y GBM pueden variar unas décimas entre computadoras.

**6. Antes y después del entrenamiento** (opcional):

```bash
python scripts/antes_despues.py
```

**7. Ver todo:** abrir `notebooks/Etapa2_Analisis.ipynb`, elegir el kernel de `.venv` y
darle **Run All**. Las Partes 1 y 2 solo necesitan `data/raw` y lo que ya está en
`results/`; la Parte 3 (SHAP) necesita `results/models.joblib`, es decir, el paso 5.

**Test 2005: ya se evaluó.** `--stage final` evalúa el test una sola vez y deja
`results/TEST_EVALUATED.lock`; con ese archivo, el script se niega a correr de nuevo.
Los resultados están en `results/test_metrics.csv`. No usar `--force`.

Para probar el código sin la tabla real: `python scripts/make_synthetic.py` y usar
`data/synthetic/dataset_synth.parquet`.

---

## Parte 1 · Datos y prevención de fuga

### Tabla 

Una fila por caballo en cada carrera, ordenada por `date`, `race_id`, `horse_no` (las
filas de una carrera quedan contiguas). La partición es cronológica por año y se puede ver de esta manera:

| split | regla | filas | carreras | período |
|---|---|---|---|---|
| train | año ≤ 2003 | 63 668 | 5 089 | 1997-06-02 → 2003-12-31 |
| val | año 2004 | 9 902 | 789 | 2004-01-04 → 2004-12-29 |
| test | año 2005 | 5 877 | 470 | 2005-01-01 → 2005-08-28 |
| **Total** | | **79 447** | **6 348** | **1997-06-02 → 2005-08-28** |

| Columna | Uso |
|---|---|
| `race_id`, `date`, `split` | identificación y partición |
| `won` | 1 si ganó; en empate por el primer lugar hay dos 1 (12 carreras) |
| `y_rel` | 3 ganó · 2 llegó 2.º–3.º · 1 llegó 4.º–6.º · 0 el resto |
| `win_odds` | cuota final |
| `horse_id`, `horse_no`, `race_no`, `result`, `place_odds` | 
| todo lo demás | features: 19 en v0; 22 en v1 |

- **v0:** `horse_age`, `horse_country`, `horse_type`, `horse_rating`, `horse_gear`,
  `declared_weight`, `actual_weight`, `draw`, `trainer_id`, `jockey_id`, `venue`,
  `config`, `surface`, `distance`, `going`, `horse_ratings`, `prize`, `race_class`,
  `field_size`.
- **v1:** v0 + `horse_win_rate`, `jockey_win_rate`, `trainer_win_rate`,
  `horse_days_since_last`, `horse_no_history` y `gear_any`, sin `jockey_id`,
  `trainer_id` ni `horse_gear` (las tasas y `gear_any` los reemplazan).

El diccionario completo de cada tabla (tipo, faltantes por split, rango y descripción)
se puede encontrar en `docs/diccionario_dataset_v0.md` y `docs/diccionario_dataset_v1.md`. Se regenera utilizando `python src/data_dictionary.py --data data/processed/dataset_v1.parquet`.

### Qué hace el código y por qué

**1. Borrado POST antes del join.** Cada CSV se limpia al leerse (`load_races`,
`load_runs`), antes de unirlos por `race_id`. Se eliminan `finish_time`,
`lengths_behind`, `position_sec*`, `behind_sec*`, `time*`, `sec_time*`, `*dividend*` y
`*combination*`. Es especialmente importante en `races.csv`: sin este paso, el join
copiaría dividendos y combinaciones ganadoras a cada corredor (`win_combination1` es
literalmente el número del ganador). El código verifica que no quede ninguna antes y
después del merge. `result` se conserva solo para construir los targets.

**2. Preparación.** No se borran filas: quitar un corredor cambiaría el tamaño del
campo. Los valores imposibles se anulan (*NaN*):

- `declared_weight < 850` (1 fila): 693 lb en la carrera 2287, de un caballo cuyo peso
  medio en sus demás carreras es 1101 lb.
- `draw > 14` (1 fila): Hong Kong tiene como máximo 14 cajones.
- Que `draw` sea mayor que el número de corredores no es un error: si un caballo se
  retira, los demás conservan su cajón (~400 filas).

Además se agrega `field_size` (corredores por carrera), y `jockey_id` y `trainer_id` se
guardan como texto porque son categorías.

**3. Targets y partición.** `won` y `y_rel` se calculan desde `result`. La partición es
cronológica por año, así que ninguna carrera queda en dos splits; el código lo verifica,
junto con `max(fecha train) < min(fecha val) < min(fecha test)`.

**4. Historial (v1).** Las tasas de victoria previas de caballo, jinete y entrenador usan
solo carreras de fechas anteriores (las del mismo día tampoco cuentan), con
*shrinkage* hacia la tasa de train: `tasa = (victorias + k · prior) / (carreras + k)`.
*k* se elige por entidad en validación, nunca en test. Más detalles en [`docs/historial_v1.md`](docs/historial_v1.md).

**5. Pruebas contra fuga.** `tests/` comprueba que no quede ninguna columna POST, que la
partición sea correcta y que, al alterar resultados a partir de una fecha, las features
anteriores no cambien. Las pruebas de fuga más a detalle se pueden encontrar en: 
[`docs/particion_y_fuga.md`](docs/particion_y_fuga.md).

### Observaciones de los datos

- `prize` falta en 462 carreras (5 849 filas), todas entre 1997-07-20 y 1998-03-18: es un
  bloque por período de recolección, no aleatorio. Los modelos lineales lo imputan con la
  mediana de su `race_class` en train.
- La carrera 3676 tiene 11 filas pero sus posiciones llegan a 12: falta el corredor que
  llegó 8.º. Ganador y targets no se ven afectados.
- `race_class` es un código (0, 6 y 11–13 son carreras especiales o de grupo), no una
  escala ordenada.
- `win_odds` está limitado a 99.
- La partición por año calendario corta a mitad de temporada (la de Hong Kong va de
  septiembre a julio), y el test (2005) es un año incompleto.

---

## Parte 2 · Modelos, métricas y calibración

### Código

| Archivo | Qué hace |
|---|---|
| `src/hkr/data.py` | Carga la tabla y valida el contrato (splits, carreras contiguas, columnas prohibidas) |
| `src/hkr/preprocess.py` | Imputación, escalado y one-hot dentro de pipelines ajustados solo con train |
| `src/hkr/models.py` | La escalera: random, market, logreg, gbm_binary, clogit, lambdamart |
| `src/hkr/metrics.py` | Top-1, MRR, NDCG@k, AUC intra-carrera, log loss, Brier, ECE, confiabilidad |
| `src/hkr/calibration.py` | Softmax por carrera + temperatura ajustada en val |
| `src/hkr/tuning.py` | Optuna con *forward chaining* dentro de train |
| `src/hkr/analysis.py` | Curvas de aprendizaje y análisis de errores |

### La escalera de modelos

Cada escalón agrega una sola idea respecto al anterior:

| Modelo | Qué aísla |
|---|---|
| `random` | piso: ¿el modelo aprende algo? |
| `market` | referencia externa (cuotas finales); nunca es feature |
| `logreg` | baseline razonable: gana / no gana, fila por fila |
| `gbm_binary` | lo mismo con no linealidad (LightGBM) |
| `clogit` | + las probabilidades de una carrera suman 1 (logit condicional de McFadden) |
| `lambdamart` | + aprende del orden completo de llegada (modelo propuesto) |

### Contrato de la tabla

`src/hkr/data.py` exige `race_id`, `date`, `split`, `won`, `y_rel` y `win_odds`. Toda
columna que no esté en `RESERVED` se toma como feature. Si aparece algo como
`finish_time`, `position_sec1` o `*_dividend`, el script se detiene con error. Para fijar
la lista de features a mano, utilizar `"features": [...]` en `configs/experiment.json`.

### Decisiones documentadas

**Métricas por carrera, no por fila.** El campo va de 4 a 14 caballos; promediar por fila
pesa más las carreras grandes, y la pregunta es quién gana la carrera. *Accuracy* por
fila está descartada: no predecir nunca un ganador da 92 %.

**MRR como objetivo del tuning**, no top-1: top-1 es 0/1 por carrera y es ruidoso; MRR
también premia dejar al ganador segundo o tercero.

**Forward chaining dentro de train** (valida en 2001, 2002 y 2003). Un k-fold barajado
dejaría que carreras futuras informen al pasado. Val (2004) no se usa para tunear, así
queda limpia para comparar modelos y calibrar. Detalle en
[`docs/busqueda_hiperparametros.md`](docs/busqueda_hiperparametros.md).

**Todos los modelos con hiperparámetros se tunean igual que LambdaMART.** Tunear solo el
modelo propuesto inflaría su ventaja.

**El número de árboles sale de *early stopping* en los folds**, y el modelo final se
reentrena con ese número fijo. Así val no se usa para *early stopping*.

**Calibración por temperatura en val.** Un solo parámetro: no cambia el orden (top-1, MRR
y NDCG quedan idénticos), solo log loss, Brier y ECE. LambdaMART no da probabilidades;
sin este paso su log loss no significa nada.

**Logit condicional sin intercepto:** cualquier constante se cancela dentro de la carrera.

**Sin class weighting ni SMOTE:** el desbalance es estructural, hay solamente un ganador por carrera y la normalización por carrera lo absorbe.

**El azar reporta el top-1 esperado exacto** promedio de 1/n no una tirada.

**Intervalos de confianza por bootstrap de carreras.** Con ~770 carreras por año, el
error estándar de top-1 es ~1.5 puntos. Cada fila trae `top1_lo/hi` (IC 95 %) y la
diferencia pareada contra el mercado con su IC. Si el IC de la diferencia incluye 0, no
se puede afirmar que el modelo le gana al mercado.

### Tablas e información importante en `results/`

- `val_metrics.csv`, `test_metrics.csv`: la tabla comparativa
- `tuning/*_trials.csv`, `tuning/*_best.json`: espacio, todas las pruebas y el tiempo
- `learning_curve_*.csv`: train contra val al variar los años de historia
- `val_errors_*`: top-1 por tamaño de campo, por cuota del ganador y según coincida o no con el favorito
- `val_confident_misses_*`, `val_missed_winners_*`: casos límite para discutir
- `val_reliability_*`, `test_reliability_*`: datos del *reliability diagram*
- `literature_reference.csv`: Silverman y Chavan, con la nota de comparabilidad
- `shap_importance_lambdamart.csv` y `figures/shap_*.png`: explicabilidad (Parte 3)

### ¿Cuánto aporta el historial? (validación 2004, top-1)

| Modelo | v0 (sin historial) | v1 (con historial) |
|---|---|---|
| Regresión logística | 18.4 % | 23.4 % |
| Logit condicional | 18.4 % | 22.8 % |
| GBM binario | 18.6 % | 19.8 % |
| LambdaMART | 21.4 % | 20.3 % |

El historial ayuda mucho a los modelos simples y casi nada a LambdaMART, que en v0 ya
sacaba esa información de los códigos de jinete y entrenador.

---

## Parte 3 · Explicabilidad (SHAP)

SHAP reparte el puntaje que LambdaMART da a cada caballo entre sus 22 variables. Se usa
el SHAP exacto para árboles que trae LightGBM (`pred_contrib=True`), sobre validación
2004. Se reporta por familia de variables y por variable, nunca por jinete o entrenador
individual. Hallazgos principales:

- **El historial domina** (aprox. 46 % de la importancia) y `jockey_win_rate` es la variable
  número uno (aprox. 21 %).
- **Las direcciones tienen sentido:** cajón cerca del riel ayuda; mucho tiempo sin correr
  y más edad restan.
- **No hay señales de fuga:** las variables de la carrera (premio, clase, distancia…)
  suman aprox. 7.5 %.
- **`gear_any` tiene drift:** el registro de equipo pasa de 0 % de los caballos en 1997
  a aprox. 80 % en 2004–2005, y la ventaja de llevarlo desaparece. Quitarla no da una mejora
  significativa por lo que el equipo decidió mantenerla.

---

## Documentación

| Documento | Contenido |
|---|---|
| [`notebooks/Etapa2_Analisis.ipynb`](notebooks/Etapa2_Analisis.ipynb) | Cuaderno único de la etapa |
| [`docs/particion_y_fuga.md`](docs/particion_y_fuga.md) | Partición, variables descartadas por fuga y pruebas |
| [`docs/historial_v1.md`](docs/historial_v1.md) | Features históricas, *shrinkage* y elección de *k* |
| [`docs/busqueda_hiperparametros.md`](docs/busqueda_hiperparametros.md) | Espacio de búsqueda, protocolo y costo |
| [`docs/paper/`](docs/paper/) | Artículo en LaTeX |
| [`DECLARACION_IA.md`](DECLARACION_IA.md) | Declaración de uso de IA |
