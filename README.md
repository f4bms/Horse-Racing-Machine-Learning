# Horse Racing & Machine Learning — Parte A: datos y prevención de fuga

Proyecto semestral de Inteligencia Artificial (TEC), Track A: ML clásico sobre
datos tabulares de carreras de Hong Kong (HKJC, 1997–2005).

Esta rama (`modelado`) contiene la **Parte A** de la Etapa 2: construir la tabla
caballo-carrera que consumen los modelos, sin información posterior a la carrera.
La Parte B (modelos, métricas, tuning) está en la rama `modelos-y-evaluacion`.

---

## Estructura

```text
.
├── scripts/
├── src/
│   ├── build_dataset.py    # genera data/processed/dataset_v0.parquet
│   └── cleaning.py         # carga, borrado POST, prepare, targets, split
├── data/
│   ├── raw/                # races.csv y runs.csv originales
│   └── processed/          # salida: dataset_v0.parquet (y .csv)
└── requirements.txt        # libreria pip usadas para ejecutar el código
```

## Cómo ejecutarlo

1. **Datos crudos.** Si hace falta, descargar `races.csv` y `runs.csv` de
   <https://www.kaggle.com/datasets/gdaley/hkracing/data> y copiarlos a `data/raw/`.

2. **Entorno** (Python ≥ 3.10):

   ```bash
   uv venv && uv pip install -r requirements.txt
   # o bien: python -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

3. **Generar la tabla:**

   ```bash
   .venv/bin/python src/build_dataset.py --csv
   ```

   Opciones: `--raw-dir` (por defecto `data/raw`), `--out` (por defecto
   `data/processed/dataset_v0.parquet`), `--csv` (guarda también una copia `.csv`).

   El script imprime filas, carreras, conteo por split y celdas que vale la pena revisar
   por columna. Si alguna validación falla, se detiene con `DataError`.

Salida esperada:

| split | regla | filas | carreras | período |
|---|---|---|---|---|
| train | año ≤ 2003 | 63 668 | 5 089 | 1997-06-02 → 2003-12-31 |
| val | año 2004 | 9 902 | 789 | 2004-01-04 → 2004-12-29 |
| test | año 2005 | 5 877 | 470 | 2005-01-01 → 2005-08-28 |
| **Total** | | **79 447** | **6 348** | **1997-06-02 → 2005-08-28** |

## Cómo usar la tabla

Una fila por caballo en cada carrera, ordenada por `date`, `race_id`, `horse_no`
(las filas de una carrera quedan contiguas).

| Columna | Uso |
|---|---|
| `race_id`, `date`, `split` | identificación y partición |
| `won` | 1 si ganó; en empate por el primer lugar hay dos 1 (12 carreras) |
| `y_rel` | 3 ganó · 2 llegó 2.º–3.º · 1 llegó 4.º–6.º · 0 el resto |
| `win_odds` | cuota final; solo para el baseline de mercado, nunca como feature |
| `horse_id`, `horse_no`, `race_no`, `result`, `place_odds` | se conservan, pero no son features |
| todo lo demás | features (19 en v0) |

Features de v0: `horse_age`, `horse_country`, `horse_type`, `horse_rating`,
`horse_gear`, `declared_weight`, `actual_weight`, `draw`, `trainer_id`,
`jockey_id`, `venue`, `config`, `surface`, `distance`, `going`,
`horse_ratings`, `prize`, `race_class`, `field_size`.

## Qué hace el código y por qué

**1. Borrado POST antes del join.** Cada CSV se limpia al leerse (`load_races`,
`load_runs`), antes de unirlos por `race_id`. Se eliminan `finish_time`,
`lengths_behind`, `position_sec*`, `behind_sec*`, `time*`, `sec_time*`,
`*dividend*` y `*combination*`. Es especialmente importante en `races.csv`,
porque sin este paso el join copiaría dividendos y combinaciones ganadoras a
cada corredor (`win_combination1` es literalmente el número del ganador). El código
verifica que no quede ninguna antes y después del merge.

`result` se conserva solo para construir los targets y nunca entra como
variable al modelo. La columna original `won` se descarta y se recalcula.

**2. Preparación.** No se borran filas: quitar un corredor cambiaría el
tamaño del campo y la estructura de un ganador por carrera. Los valores
imposibles se anulan (*NaN*).

- Se agrega `field_size`: cuántos corredores tiene cada carrera.
- `jockey_id` y `trainer_id` se guardan como texto: son categorías, no
  cantidades.
- Valores fuera de rango que se anulan a *NaN*:
  - `declared_weight <850` (1 fila): 693 lb en la carrera 2287, caballo 2931,
    cuyo peso medio en sus demás carreras es 1101 lb. Los siguientes valores
    más bajos (896 y 898 lb) son coherentes con el historial de su caballo y
    se conservan.
  - `draw > 14` (1 fila): draw 15 en la carrera 5188 (`horse_no` 3). Hong Kong
    tiene como máximo 14 cajones. Por eso `draw` queda como *NaN*.
- Que `draw` sea mayor que el número de corredores no es un error: si un
  caballo se retira, los demás conservan el cajón asignado (pasa en ~400 filas).
- `horse_gear = "--"` significa "sin equipo" y se conserva como categoría
  válida (por ahora).

**3. Targets y partición.** `won` y `y_rel` se calculan desde `result`
(`won` coincide al 100 % con la columna original). La partición es
cronológica por año, así que ninguna carrera queda en dos splits. El código lo
verifica, junto con `max(fecha train) < min(fecha val) < min(fecha test)` y que
toda carrera tenga al menos un ganador.

### Observaciones de los datos (posibles consideraciones)

- `prize` falta en 462 carreras (5 849 filas), todas entre 1997-07-20 y
  1998-03-18. Es un bloque por período de recolección, no aleatorio.
- La carrera `3676` (2002-02-20) tiene 11 filas, pero sus posiciones llegan
  a 12: falta el corredor que llegó 8.º. Ganador y targets no se ven afectados.
- Hay 1 fila con `draw > 14`, por encima del número de cajones; podría ser
  un error de registro o se retiro el competidor, por eso se cambio a *NaN*.
- `race_class` es un código (0, 6, 11–13 son carreras especiales/de grupo
  según `horse_ratings = "G"`), así que no es una escala ordenada.
- `win_odds` está limitado a 99.0.
- La partición por año calendario corta a mitad de temporada (la temporada de
  HK va de septiembre a julio), y test (2005) es un año incompleto.

---

### Pendiente
- Target encoding fuera de fold de `jockey_id` / `trainer_id`
- `dataset_v1` (con historial)
<!-- Completar cuando se implemente:
     - lista de features históricas y su definición
     - valor de m del shrinkage y por qué
     - cómo se garantiza que solo se usan fechas anteriores
     - Features históricas (ventana expansiva + `shift(1)`) y shrinkage para debutantes
-->
- Pipeline (scikit-learn: imputación, escalado y codificación, ajustado solo con train) y tests contra fuga (pytest)
<!-- Completar: archivo del pipeline, qué transforma cada rama
     (numéricas / categóricas), y cómo correr los tests. -->
- Documento de justificación (cronológica vs. aleatoria, proporciones, variables descartadas)
<!-- Enlazar aquí docs/... cuando exista. -->

