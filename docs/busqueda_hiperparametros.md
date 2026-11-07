# Búsqueda de hiperparámetros: espacio, protocolo y costo

Este documento justifica cómo se ajustaron los hiperparámetros de la parte de
modelado. Los números exactos de cada corrida (mejores valores, puntaje en
validación cruzada, segundos) **no se copian aquí a mano**: salen de
`results/tuning/*_best.json` y `*_trials.csv`, y el notebook
`notebooks/02_modelos_y_resultados.ipynb` los muestra en tablas.

El código está en `src/hkr/tuning.py` (`SEARCH_SPACE`).

---

## 1. Método: Optuna (TPE) y no una grilla

En clase se vio `GridSearchCV`. Aquí se usa **Optuna con el muestreador TPE**,
que el enunciado permite explícitamente ("Optuna, grid search o random search").

- Los modelos de árboles tienen **6 hiperparámetros continuos o casi continuos**.
  Una grilla de apenas 4 valores por eje serían 4⁶ = 4 096 combinaciones; con
  3 folds, más de 12 000 entrenamientos.
- TPE usa las pruebas anteriores para decidir dónde probar la siguiente, así que
  llega a una zona buena con **40 pruebas**. La historia de la búsqueda (notebook,
  sección 2) muestra si la mejora ya se había aplanado antes de la última prueba.
- Con **semilla fija** (42) la búsqueda es reproducible. Cada prueba queda
  registrada en `results/tuning/<modelo>_trials.csv`: valores probados,
  puntaje, desviación y duración.

## 2. Protocolo: validación cruzada *forward chaining* dentro de train

| Fold | Entrena con | Valida en |
|---|---|---|
| 1 | 1997 – 2000 | 2001 |
| 2 | 1997 – 2001 | 2002 |
| 3 | 1997 – 2002 | 2003 |

- **Nunca se baraja.** Un k-fold aleatorio dejaría que carreras futuras
  informen al modelo sobre el pasado (Bunker y Thabtah, 2019, lo desaconsejan
  explícitamente en predicción deportiva).
- **Ni 2004 ni 2005 participan.** 2004 (validación) queda limpio para comparar
  modelos y calibrar. 2005 (test) se usa una sola vez al final.
- **Objetivo: MRR medio de los 3 folds.** Se prefiere a top-1 porque top-1 vale
  0 o 1 por carrera y es muy ruidoso; MRR también premia dejar al ganador 2.º o
  3.º. Se guarda la media **y la desviación estándar** de los folds.
- **Número de árboles:** en cada fold se usa *early stopping* (100 rondas sin
  mejorar) sobre el año de validación de ese fold. El modelo final se reentrena
  con todo train y con el promedio de esas iteraciones, **fijo**. Así 2004 no
  se usa para decidir cuándo parar.
- **Todos los modelos con hiperparámetros se tunean igual**, no solo el
  propuesto. Tunear solo LambdaMART inflaría su ventaja sobre los baselines.

## 3. Espacio de búsqueda y por qué

### Árboles (`gbm_binary` y `lambdamart`, LightGBM) — 40 pruebas cada uno

| Hiperparámetro | Rango | Escala | Por qué |
|---|---|---|---|
| `learning_rate` | 0.01 – 0.2 | log | Por debajo de 0.01 necesita miles de árboles; por encima de 0.2 el boosting se vuelve inestable. Escala log porque importa la proporción, no la diferencia |
| `num_leaves` | 7 – 127 | log | Controla la complejidad de cada árbol. 7 hojas ≈ profundidad 3; 127 ≈ profundidad 7. Con ~22 features no hace falta más |
| `min_child_samples` | 3 – 300 | log | Mínimo de filas por hoja: el freno principal contra el sobreajuste. El campo promedio tiene ~12 caballos, así que 300 equivale a ~25 carreras |
| `colsample_bytree` | 0.4 – 1.0 | lineal | Fracción de features por árbol; menos de 0.4 dejaría fuera demasiadas con solo ~22 |
| `subsample` | 0.5 – 1.0 | lineal | Fracción de filas por árbol (*bagging* dentro del boosting) |
| `reg_lambda` | 0.001 – 30 | log | Regularización L2 de los pesos de las hojas |

Fijos: `subsample_freq = 1`, `n_estimators` hasta 2 000 con *early stopping*.

### Lineales — 15 pruebas cada uno (un solo hiperparámetro)

| Modelo | Hiperparámetro | Rango | Por qué |
|---|---|---|---|
| `logreg` | `C` (inverso de la regularización L2) | 10⁻⁵ – 10 | La regularización además resuelve la colinealidad `horse_rating` / `actual_weight` (Tabla V del paper) |
| `clogit` | `alpha` (penalización L2) | 10⁻⁶ – 1 | Mismo papel que `C`, en la escala de la función de pérdida del logit condicional |

Con un solo hiperparámetro, 15 pruebas cubren el rango de sobra.

### Revisión de bordes (corrección hecha)

El notebook revisa automáticamente si algún hiperparámetro elegido quedó en el
5 % extremo de su rango, porque un óptimo pegado al borde puede indicar que el
mejor valor está afuera.

- **`logreg`, `C`:** en la primera corrida eligió `C ≈ 0.00102`, pegado al
  mínimo original (0.001). Se amplió el rango a 10⁻⁵ y, al volver a correr,
  eligió `C ≈ 0.002`, ya en el interior. Corregido.
- **`lambdamart` y `gbm_binary`, `min_child_samples`:** con el mínimo original
  (10) quedaron cerca del borde. Se bajó el mínimo a 3 y volvieron a elegir
  valores bajos (3 y 6). Aquí **no se siguió ampliando**, porque la búsqueda
  muestra que este hiperparámetro es plano: en LambdaMART, el mejor MRR con
  hojas de 3–10 filas (0.406) y con hojas de 30–100 filas (0.405) difieren en
  0.001, diez veces menos que la desviación entre folds (≈ 0.014). El modelo
  se regulariza por otras vías (árboles chicos, 8–13 hojas, y *early
  stopping*), así que el tamaño mínimo de hoja casi no cambia el resultado. El
  mínimo absoluto posible es 1, muy cerca de 3.

## 4. Costo de cómputo

El costo por modelo (segundos totales y por prueba) queda en
`results/tuning/<modelo>_best.json` y lo resume el notebook. Como referencia,
en la corrida de desarrollo (2 núcleos de CPU, sin GPU) **la búsqueda completa
de los cuatro modelos tomó del orden de 4 a 5 minutos**, y la corrida completa
con curvas de aprendizaje, unos 10 minutos.

**Por qué este presupuesto es suficiente:**

- La historia de la búsqueda muestra que el mejor puntaje deja de mejorar mucho
  antes de la última prueba.
- La diferencia entre las mejores configuraciones es menor que la desviación
  estándar entre folds (entre 0.013 y 0.020 de MRR según el modelo). Seguir
  buscando optimizaría ruido, no el modelo.
- Más pruebas aumentarían el riesgo de sobreajustar la propia búsqueda a los
  tres años de los folds.

## 5. Lo que la búsqueda no puede arreglar

Las curvas de aprendizaje (notebook, sección 3) muestran que LambdaMART
mantiene una brecha grande entre train y validación aun con los mejores
hiperparámetros, mientras la regresión logística no la tiene. Eso indica que
la limitación no está en el ajuste fino sino en la **cantidad de señal** que
tienen las features antes de la carrera, frente a lo que ya sabe el mercado.
