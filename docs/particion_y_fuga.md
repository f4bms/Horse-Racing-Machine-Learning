# Partición de datos y variables descartadas por fuga

Este documento responde a:

- Cómo se particionó y con qué proporciones.
- Si la partición es aleatoria o cronológica, y por qué.
- Qué variables se descartaron por riesgo de *target leakage*.

Cada regla está implementada en `src/cleaning.py` y verificada por un test en `tests/`.

Se marca explícitamente lo que es **supuesto** (no verificable con los datos disponibles) y lo que está **verificado**.

---

## 1. Problema de predicción y momento de predicción

- **Unidad:** una fila por caballo en cada carrera (79 447 filas, 6 348 carreras, 1997-06-02 -> 2005-08-28).
- **Momento de predicción:** antes de que empiece la carrera. Cada columna cumple una de tres reglas:
  - **Features** (entradas del modelo): solo son admisibles si su valor se conoce **antes de la salida**. Toda información generada durante o después de la carrera se descarta.
  - **Targets** (lo que se predice): `won` (1 si llegó 1.º; en empate por el primer lugar hay dos ganadores, en 12 carreras) y `y_rel` (3 ganó, 2 llegó 2.º–3.º, 1 llegó 4.º–6.º, 0 el resto). Se conocen recién al terminar la carrera, por definición. Cada fila los calcula solo desde su propio `result` y nunca entran como feature.
  - **Features históricas** (dataset_v1): se construyen con los resultados de carreras de **fechas anteriores**; las del mismo día tampoco cuentan.

## 2. Partición

### 2.1 Regla

La partición es cronológica, por año calendario de la fecha de la carrera:

| split | regla | filas | % filas | carreras | % carreras | período | tasa de victoria |
|---|---|---|---|---|---|---|---|
| train | año ≤ 2003 | 63 668 | 80.1 % | 5 089 | 80.2 % | 1997-06-02 -> 2003-12-31 | 8.01 % |
| val | año 2004 | 9 902 | 12.5 % | 789 | 12.4 % | 2004-01-04 -> 2004-12-29 | 7.98 % |
| test | año 2005 | 5 877 | 7.4 % | 470 | 7.4 % | 2005-01-01 -> 2005-08-28 | 8.01 % |

La unidad que se asigna a un split es la **carrera**, no la fila: todos los corredores de una carrera quedan en el mismo split. Además, `max(fecha train) < min(fecha val)` y `max(fecha val) < min(fecha test)`.

### 2.2 Por qué cronológica y no aleatoria

1. **Así se usa el modelo en la práctica:** siempre se predicen carreras futuras con lo ocurrido antes. Una partición aleatoria pone en train carreras posteriores a las de test. El modelo aprendería entonces de la forma futura de caballos, jinetes y entrenadores que luego "predice", lo que es **fuga temporal**.
2. **Es requisito para las features históricas.** Las tasas de victoria de caballo, jinete y entrenador se calculan con carreras anteriores. Con una partición aleatoria, el historial de una fila de train podría incluir carreras de test y generar resultados sesgados.
3. **En las lecturas revisadas.**
   - Bunker y Thabtah (2019, *A machine learning framework for sport result prediction*), indican que en predicción deportiva hay que preservar el orden de las instancias, para que los partidos futuros se predigan solo con partidos pasados. Señalan que la validación cruzada que baraja las instancias "no es un medio apropiado" y recomiendan un *hold-out* que preserve el orden. Mencionan además como enfoque común entrenar con temporadas anteriores y evaluar en las siguientes.
   - Schumaker (2018, *Machine Learning the Harness Track*) parte sus datos de carreras en un periodo de entrenamiento de doce meses y un periodo de prueba de los tres meses siguientes.
4. **Dividir por carrera evita otra fuga:** si los corredores de una carrera quedaran en splits distintos, el modelo vería en train a los rivales y el resultado de una carrera de test.

### 2.3 Por qué esta proporción

- **Bloques de un año calendario:** cada bloque es un periodo completo y fácil de auditar (la regla es una comparación de años, sin cortes manuales).
- **Val = 2004 completo:** cubre todos los meses del calendario de carreras, así que las decisiones de modelado (hiperparámetros, k del *shrinkage*) se toman con todas las épocas del año.
- **Test = 2005:** es el año más reciente y el más cercano a "producción". Se usa una sola vez, al final.
- **Train = 6.5 años (80% aprox.):** se necesitan varios datos, y de forma robusta para tener buenos resultados con el entrenamiento del modelo y el historial necesita años previos para acumular carreras por caballo, jinete y entrenador.
- **Los tres splits tienen la misma tasa de victoria (8.0% aprox.):** no hay desviación visible del target entre periodos.

### 2.4 Limitaciones asumidas

- **El corte por año calendario parte temporadas.** La temporada de Hong Kong va de septiembre a julio, así que train termina y val empieza a mitad de la temporada 2003/04. Sin embargo, el dataset cuenta con registros de agosto, pero no de abril, por lo que tampoco es un enfoque incorrecto centrarse en el año más que en la temporada, ya que es información que no esta clara.
- **Test (2005) es un año incompleto:** solo llega a 2005-08-28, con 470 carreras. Las métricas de test tendrán más varianza que las de val.
- **`races.csv` tiene una carrera sin corredores** (`race_id` 6108). El *inner join* la descarta y no afecta a ningún split.

### 2.5 Validación cruzada

La validación cruzada consiste en partir los datos de entrenamiento en varios pedazos (*folds*), entrenar con unos, evaluar con otro y repetir. Así el resultado no depende de una sola partición.

El problema es la versión típica, *k-fold*: reparte las filas al azar. Eso pone carreras de 2003 en el entrenamiento y carreras de 1999 en la evaluación, o sea que el modelo aprende del futuro para "predecir" el pasado, esto seraí una fuga temporal solo que dentro de train.

Por eso, si se usa, tiene que respetar el tiempo. Cada pedazo se evalúa con un año que viene después de todo lo que se usó para entrenar, y el entrenamiento va creciendo, por ejemplo con `k=4`:

| Fold | Entrena con | Evalúa con |
|---|---|---|
| 1 | 1997–1999 | 2000 |
| 2 | 1997–2000 | 2001 |
| 3 | 1997–2001 | 2002 |
| 4 | 1997–2002 | 2003 |

> Los años en el ejemplos estan partidos muy convenientemente, la idea es solo respetar el hecho de que los datos de entrenamiento sean menor que los de test, no estrictamente como en el ejemplo.

Todo esto pasa dentro de train (≤ 2003). Al dividir por año, ninguna carrera queda partida entre entrenamiento y evaluación. Sirve para elegir hiperparámetros con más de una medición. Después, val (2004) confirma la elección y test (2005) se usa una sola vez, al final.

## 3. Variables descartadas por fuga (target leakage)

Todas se borran al leer cada CSV, antes del join (`load_runs` y `load_races`). En `races.csv` esto es crítico: si se borraran después del join, cada corredor heredaría los dividendos y combinaciones ganadoras de su carrera, y `win_combination1` es literalmente el número del caballo ganador.

### 3.1 `runs.csv` (20 columnas)

| Columna(s) | Por qué es POST |
|---|---|
| `finish_time` | Tiempo final del caballo: solo existe cuando termina la carrera. |
| `lengths_behind` | Distancia al ganador en la llegada. |
| `position_sec1` … `position_sec6` | Posición del caballo en cada tramo de la carrera. |
| `behind_sec1` … `behind_sec6` | Distancia al líder en cada tramo. |
| `time1` … `time6` | Tiempos por tramo del caballo. |

### 3.2 `races.csv` (26 columnas)

| Columna(s) | Por qué es POST |
|---|---|
| `sec_time1` … `sec_time7` | Tiempos por tramo de la carrera. |
| `time1` … `time7` | Tiempos acumulados de la carrera. |
| `win_combination1`, `win_combination2` | Caballo(s) ganador(es): es el target. |
| `place_combination1` … `place_combination4` | Caballos que llegaron en puestos o podio, *place*: deriva del resultado. |
| `win_dividend1`, `win_dividend2`, `place_dividend1` … `place_dividend4` | Dividendos pagados: se calculan con el resultado. |

Los patrones que reconocen estas columnas están en `FORBIDDEN_PATTERNS` (en el cleaning). Los tests comparan contra la lista explícita de arriba: si un patrón deja de reconocer una columna, o reconoce una de más, el test falla.

### 3.3 Columnas conservadas que nunca son features

`NON_FEATURES` en `src/data_dictionary.py` fija su rol, y todo lo que no está ahí es feature:

| Columna | Rol | Motivo |
|---|---|---|
| `result` | resultado | Es POST. Se conserva solo para calcular `won` y `y_rel`. |
| `won` | target | Se descarta al cargar `runs.csv` y se recalcula desde `result`. Coincide al 100 % con la original. |
| `win_odds` | mercado | Cuota final: se fija justo al cerrar las apuestas y resume la información del mercado. Se usa solo como *baseline* de mercado. |
| `place_odds` | mercado | Igual que `win_odds`. Además falta en el 0.19 % de train, el 8.81 % de val y el 46.69 % de test. |
| `race_id`, `date`, `horse_id`, `horse_no`, `race_no` | id | Identificadores: no describen al caballo ni a la carrera. |
| `split` | partición | Etiqueta de la partición. |

> `won` se recalcula desde result para no depender de la columna original y evitar contradicciones con el y_rel en caso de algun error; hay un test que valida si la columna original y la recalculada coinciden (en caso de no hacerlo se puede revisar contra dataset, pero este no fue el caso).

Crear `won` y `y_rel` no introduce fuga: cada fila los calcula solo desde su propio `result`, sin estadísticas de otras filas ni de otros splits. La fuga aparecería si el target llegara a las features. Hay dos casos que se deben de vigilar: tasas históricas sin desplazar por fecha y un *target encoding* ajustado con toda la tabla, ambos estan cubiertos en `tests/test_leakage_checks.py`. Además, las filas de cada carrera van ordenadas por `horse_no` y no por `result`, para que la posición de la fila no delate al ganador.

## 4. Features admitidas en v0 y su disponibilidad antes de la carrera

Son 19 features. La pregunta para cada feature es: **¿se conoce antes de la salida?**

| Feature | ¿PRE? | Evidencia / riesgo |
|---|---|---|
| `horse_rating` | Sí, **verificado** | Si el rating ya incluyera el resultado de la carrera, la subida de rating *hacia* esa carrera estaría asociada a ganarla. En los datos, ganar se asocia con la subida **siguiente** (correlación 0.292; +2.31 puntos de media para los ganadores) y casi nada con la **anterior** (0.052). Por lo tanto, el rating es previo a la carrera. El test exige \|anterior\| < 0.1 y siguiente > 0.2. |
| `declared_weight`, `actual_weight` | Supuesto | Pesos declarados o asignados antes de la carrera. Los valores imposibles se anulan a NaN (`declared_weight < 850`). |
| `draw` | Supuesto | Cajón de salida, sorteado antes de la carrera. Los valores > 14 se anulan a NaN. |
| `jockey_id`, `trainer_id` | Supuesto | Inscripción de jinete y entrenador, conocida antes de la carrera. |
| `horse_gear` | Supuesto | Equipo declarado del caballo. |
| `horse_country`, `horse_type` | Supuesto | Atributos fijos del caballo. `horse_type` mezcla sexo y color (ver el diccionario). |
| `horse_age` | **Riesgo abierto** | De los 1 601 caballos que corren más de 2 años, 1 033 tienen una sola edad en todas sus carreras. Probablemente es la edad en una fecha fija y no en la de cada carrera. No es fuga del target, pero es información con fecha distinta a la de la carrera. Decisión pendiente: descartarla o recalcularla (puntos 7–8). |
| `venue`, `config`, `surface`, `distance`, `race_class`, `horse_ratings`, `prize` | Supuesto | Condiciones de la carrera publicadas antes de correrla. Verificado: son constantes dentro de cada carrera. |
| `going` | Supuesto | Estado de la pista; puede actualizarse el mismo día, pero antes de la carrera. |
| `field_size` | Derivada, PRE | Número de corredores que tomaron la salida (filas por carrera). Excepción conocida: la carrera 3676 tiene una fila menos de las que indica su `result`. |

### 4.1 Control de *proxy* del resultado

Una feature con fuga "sabe" el resultado, así que se parecería a la cuota final del mercado (`win_odds`). Hay dos controles:

| Control | Aplica a | Medida | Umbral | Máximo legítimo en v0 | `win_odds` |
|---|---|---|---|---|---|
| Ordena la carrera como el resultado | numéricas | \|Spearman\| con `result` **dentro de cada carrera**, promediado | < 0.3 | `draw` 0.105 | 0.494 |
| Predice al ganador | categóricas | \|AUC − 0.5\| en val, con cada valor reemplazado por su tasa de victoria en train | < 0.25 | `jockey_id` 0.649 | 0.80 |

Decisiones:

- **Se mide dentro de la carrera** porque con todas las filas juntas las variables de nivel carrera se inflan. Por ejemplo, `field_size` daba 0.16 solo porque en carreras más grandes hay posiciones más altas, aunque no puede indicar quién gana. Las features de nivel carrera (`distance`, `prize`, etc.) son iguales para todos los corredores, así que no pueden delatar al ganador y el primer control las omite.
- **Para las categóricas se usa AUC,** porque no tienen orden. Se ajusta con train y se mide con val; test no se usa.
- **Se toma el valor absoluto,** porque una fuga puede tener cualquier signo.
- **Control positivo:** `win_odds` debe superar ambos umbrales. Si no lo hiciera, los controles no tendrían poder para detectar nada.

Son heurísticas: una feature legítima puede ser muy predictiva y una fuga débil podría no destacar. El Track A pide el mismo control con explicabilidad: "si una variable inesperada domina el modelo, con frecuencia es la firma de una fuga".

## 5. Otras reglas contra fuga y cómo se verifican

| Regla | Dónde se aplica | Test(s) |
|---|---|---|
| Borrar las columnas POST antes del join | `load_runs`, `load_races`, `merge` | `test_patrones_post_reconocen_exactamente_*`, `test_merge_rechaza_post_antes_del_join`, `test_salida_sin_columnas_post` |
| Ninguna columna cruda desaparece ni aparece sin justificar | `build_v0` | `test_toda_columna_cruda_esta_justificada` |
| Las features son exactamente las revisadas en §4 | `NON_FEATURES`, `feature_columns` (`src/data_dictionary.py`) | `test_features_de_v0_son_las_revisadas`, `test_targets_resultado_y_mercado_no_son_features` |
| Ninguna feature es un *proxy* del resultado (§4.1) | — | `test_el_mercado_supera_los_umbrales`, `test_ninguna_numerica_ordena_la_carrera_como_el_resultado`, `test_ninguna_categorica_predice_al_ganador_como_el_mercado`, `test_horse_rating_es_el_rating_previo_a_la_carrera` |
| El orden de las filas no depende del resultado | `sort_by_race` | `test_orden_de_filas_no_depende_del_resultado` |
| Partición cronológica, por carrera, sin cruces | `add_split`, `validate` | `test_split_sigue_la_regla_por_anio`, `test_ninguna_carrera_cruza_splits`, `test_splits_en_orden_cronologico_estricto`, unitarios de `validate` |
| Nada se ajusta con la tabla completa (sin imputar, escalar ni codificar antes de particionar) | `prepare` | `test_valores_iguales_a_los_crudos_salvo_correcciones`, `test_no_se_pierden_corredores` |
| Las features históricas solo usan fechas **anteriores** (las del mismo día tampoco) | dataset_v1 (punto 7) | `assert_uses_only_past` en `tests/test_leakage_checks.py`. Hoy se prueba con un dataset de juguete: acepta las tasas previas correctas y detecta tres fugas (incluir la carrera actual, usar carreras del mismo día, usar todo el historial). |
| El preprocesamiento se ajusta solo con train; test se usa una vez al final; k y los hiperparámetros se eligen en val | Pipeline de la Parte B | Pendiente en la Parte B |

Para correr los tests (desde la raíz del repositorio):

```bash
.venv/bin/python -m pytest
```

### 5.1 Verificación de que los tests detectan fugas reales

Se sabotearon tres veces, sobre una copia temporal, las funciones de validación (safeguard) de `cleaning.py`, y los tests fallaron en los tres casos:

| Sabotaje | Tests que fallan |
|---|---|
| Quitar el patrón `dividend` (cols) | 4 (unitarios de patrones y lista explícita de `races.csv`) |
| Dejar pasar `lengths_behind` como feature | 4 (patrones, lista explícita, columnas justificadas y features revisadas) |
| Imputar `prize` con la media de toda la tabla | `test_valores_iguales_a_los_crudos_salvo_correcciones` |

También se agregaron a la tabla dos features con fuga plantada, y los controles con el *proxy* las detectaron:

| Feature plantada | Medida | Test que falla |
|---|---|---|
| `result` + ruido (σ = 4) | \|Spearman\| en carrera = 0.676 | `test_ninguna_numerica_ordena_la_carrera_como_el_resultado` |
| "llegó al top 3" como texto (`A`/`B`) | AUC = 0.913 | `test_ninguna_categorica_predice_al_ganador_como_el_mercado` |

Cada sabotaje representa un tipo distinto de fuga (columnas POST e imputación con toda la tabla), y todos hacen fallar al menos un test. La imputación es la más difícil de notar, porque no agrega columnas sospechosas, y solo la detecta la comparación contra los datos originales. Las fugas plantadas superan al propio mercado aunque tengan ruido (0.676 frente a 0.494 y AUC 0.913 frente a 0.80), así que una fuga real difícilmente pasaría desapercibida. Lo que los tests no garantizan es la ausencia de una fuga débil, por debajo de los umbrales: para eso está la revisión de cada feature y, más adelante con SHAP, la explicabilidad del modelo.
