# Features históricas (dataset_v1)

dataset_v1 es dataset_v0 con 6 columnas nuevas (5 de historial y `gear_any`), calculadas en `src/history.py`, y sin `jockey_id`, `trainer_id` ni `horse_gear`, que quedan reemplazadas (§1.1). Todo lo demás queda igual: las mismas filas, valores, índice y partición (hay un test que lo comprueba).

```bash
.venv/bin/python src/build_dataset.py --version v1 --csv
.venv/bin/python src/data_dictionary.py --data data/processed/dataset_v1.parquet
```

## 1. Qué columnas agrega

| Columna | Qué es | Si no hay historial |
|---|---|---|
| `horse_win_rate` | Tasa de victoria previa del caballo, con *shrinkage* | Vale el prior (8.01 %) |
| `jockey_win_rate` | Tasa de victoria previa del jinete, con *shrinkage* | Vale el prior |
| `trainer_win_rate` | Tasa de victoria previa del entrenador, con *shrinkage* | Vale el prior |
| `horse_days_since_last` | Días desde la carrera anterior del caballo | NaN (lo trata el pipeline) |
| `horse_no_history` | 1 si es la primera carrera del caballo en el dataset | — |
| `gear_any` | 1 si lleva algún equipo (no es historial: reemplaza a `horse_gear`) | — |

### 1.1 Columnas de v0 que v1 reemplaza

| Columna de v0 | En v1 se usa | Por qué |
|---|---|---|
| `jockey_id` | `jockey_win_rate` | La tasa resume lo que el ID aporta y se actualiza con cada carrera. |
| `trainer_id` | `trainer_win_rate` | Igual que el jinete. |
| `horse_gear` | `gear_any` | Son 822 combinaciones de equipo con muy poca señal. Lo único que muestra una diferencia es si lleva algún equipo o no. |

Las columnas de v0 se **quitan de la tabla**, no solo se marcan, porque el código de modelos toma como feature toda columna que no esté en su lista de excluidas. Si hace falta recuperarlas, v0 y v1 tienen las mismas filas con el mismo índice, y en caso de haberse reordenado se puede agrupar por `race_id` y `horse_no`.

**Por qué la tasa histórica y no el ID o un *target encoding* clásico.** Para el jinete y el entrenador había tres opciones:

- **Usar el ID tal cual.** El modelo aprende un valor por jinete. El problema es que en test tiene algunos jinetes que no aparece en train, y de esos no sabe nada.
- ***Target encoding* clásico.** Se reemplaza cada ID por su tasa de victoria en train. El problema es que el valor queda fijo, así que un jinete que debuta en 2005 tiene el promedio todo el año, aunque gane mucho. Además, si dentro de train se calcula con *folds* al azar, mezcla carreras futuras y ocasiona una fuga.
- **Tasa histórica (la elegida).** También es un *target encoding*, pero se recalcula en cada fecha con las carreras anteriores. Los jinetes nuevos empiezan en el promedio y suben o bajan a medida que corren.

## 2. Cómo se garantiza que solo usan el pasado

Cada fila usa solo carreras con **fecha anterior** a la suya. Las del mismo día tampoco cuentan: un jinete corre varias carreras por jornada, y sin esta regla su primera carrera podría usar el resultado de la tercera.

Para eso, las carreras y victorias se suman por (entidad, fecha) y se desplazan una fecha. Un `shift(1)` por fila no sirve, porque deja pasar las carreras del mismo día.

Se verifica en `tests/test_leakage_checks.py`. Se alteran los resultados desde cuatro fechas de corte (cercano al inicio de train y mitad del dataset, primer día de val y primer día de test) y se comprueba que las features anteriores al corte no cambian.

Las filas de val y test sí usan el historial de carreras de val y test anteriores a ellas. Cuando se predice una carrera de 2005, las carreras de 2004 y de los días previos de 2005 ya ocurrieron, por lo que no puede considerarse fuga.

## 3. *Shrinkage* y elección de k

Un caballo que ganó su única carrera tiene una tasa de 100 %, que no es creíble. El *shrinkage* la acerca a la tasa general según cuántas carreras tenga:

```
tasa = (victorias_previas + k · prior) / (carreras_previas + k)
```

- **prior** = tasa de victoria en train = 0.0801. Se calcula solo con train.
- **k** controla cuánto se confía en la tasa propia. Con k = 100, un caballo necesita unas 100 carreras para que su tasa pese tanto como el prior.

k se elige **por separado para cada entidad** (columna), probando el grid `0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000`. Se queda el que da menor *log loss* en validación **(val)**, usando la tasa como probabilidad de ganar. La parte de test no se usa. La tabla completa se guarda en `docs/seleccion_k_dataset_v1.csv`.

| Entidad | k elegido | *log loss* en val: sin *shrinkage* (k = 0) → con k elegido |
|---|---|---|
| Caballo | **100** | 0.5628 → 0.2775 |
| Jinete | **50** | 0.2746 → 0.2648 |
| Entrenador | **500** | 0.2783 → 0.2760 |

Sin *shrinkage*, la tasa del caballo es muy mala (0.56), porque la mayoría tiene pocas carreras. Para el jinete y el entrenador la mejora es menor, porque acumulan muchas carreras.

## 4. Controles

| Qué se comprueba | Test |
|---|---|
| Solo usan fechas anteriores (datos reales) | `test_features_historicas_de_v1_usan_solo_el_pasado` |
| v1 = v0 + 6 columnas − 3 reemplazadas, sin cambiar nada más | `test_v1_es_v0_mas_el_historial` |
| `jockey_id`, `trainer_id` y `horse_gear` no están en v1 | `test_columnas_reemplazadas_no_estan_en_v1` |
| Lo quitado se recupera de v0 por índice | `test_columnas_reemplazadas_se_recuperan_de_v0_por_indice` |
| El prior y los k no cambian si se alteran los resultados de test | `test_ajuste_del_historial_no_mira_test` |
| "Sin historial" = primera carrera = días NaN | `test_sin_historial_es_la_primera_carrera_del_caballo` |
| Ninguna feature nueva ordena la carrera como el resultado (\|Spearman\| < 0.3) | `test_ninguna_feature_nueva_ordena_la_carrera_como_el_resultado` |

Correlación de Spearman dentro de la carrera con `result`:

| `jockey_win_rate` | `horse_no_history` | `trainer_win_rate` | `horse_days_since_last` | `horse_win_rate` |
|---|---|---|---|---|
| −0.229 | 0.147 | −0.109 | 0.034 | −0.030 |

Como en v0, se sabotearon tres veces las protecciones, sobre una copia de `history.py`, y los tests fallaron en los tres casos:

| Sabotaje | Tests que fallan |
|---|---|
| Quitar el desplazamiento por fecha (incluye la carrera actual) | `test_features_historicas_de_v1_usan_solo_el_pasado`, `test_sin_historial_es_la_primera_carrera_del_caballo` |
| Calcular el prior con toda la tabla | `test_ajuste_del_historial_no_mira_test` |
| Elegir k en test en vez de val | `test_ajuste_del_historial_no_mira_test` |

## 5. Aspectos a considerar

- **La tasa del caballo aporta poco** (−0.03 dentro de la carrera), y su k es alto. Probablemente `horse_rating` ya resume la habilidad del caballo. La del jinete es la más informativa (−0.229), aunque sigue lejos del umbral de 0.3 y del mercado (0.494).
- **El k del entrenador es casi indiferente.** La curva es plana: entre 200 y 1000 el *log loss* cambia en el quinto decimal. Su tasa aporta poca información.
- **El criterio para elegir k es simple.** Usa la tasa como si fuera la probabilidad de ganar, sin tener en cuenta cuántos rivales hay en la carrera; elige el k que mejor estima la tasa de cada entidad.
- **Usar val para elegir k** es lo mismo que usarla para elegir hiperparámetros. Por eso test sigue siendo la única medición limpia del final.
- **Sacar los IDs puede perder patrones propios de cada jinete, entrenador y equipamiento** que la tasa no captura (por ejemplo, rendir distinto según la pista de carrera). Se ve al comparar v0 (con IDs) contra v1 (con tasas), si hiciera falta.
- **`horse_days_since_last`** llega hasta 1 099 días (son algo frecuentes los descansos largos) y es NaN en el debut. No se imputa: lo trata el pipeline con train.
