# Declaración de uso de herramientas de IA — Etapa 2

Declaramos cómo usamos herramientas de IA en esta entrega, en qué partes, con qué propósito y qué verificamos manualmente.
## Jorge Gutiérrez — Modelos, métricas y evaluación

**Herramienta:** Claude (Anthropic), usado como asistente de programación.

**En qué partes y para qué:**
- Generación y depuración de código para el módulo de modelado (`src/hkr/`, `scripts/run_experiments.py` y `scripts/antes_despues.py`), a partir de lo que planteamos en la metodología de la Etapa 1: escalera de modelos, métricas por carrera, calibración y búsqueda de hiperparámetros.
- Apoyo en la redacción del notebook `02_modelos_y_resultados.ipynb` y de la documentación (`docs/busqueda_hiperparametros.md`, `docs/paper/metodologia_modelado.tex`).
- Explicación de conceptos (métricas de ranking, validación cruzada *forward chaining*, calibración, intervalos por bootstrap) y apoyo con Git.

**Qué hicimos y verificamos nosotros:**
- Ejecutamos todo el pipeline, entrenamos los modelos y corrimos el test de 2005 una sola vez, después del análisis SHAP.
- Corrimos las 89 pruebas automáticas y todas pasan.
- Reprodujimos el entrenamiento en dos computadoras y comparamos los resultados.
- Comprobamos que los baselines reproducen los valores del paper de la Etapa 1 (mercado 29.8 % y azar 8.1 %).
- Comprobamos que los modelos sin entrenar aciertan lo mismo que el azar (64 de 789 carreras) y entrenados entre 156 y 185.
- Revisamos los hiperparámetros en el borde de su rango y ampliamos el de la regresión logística.
- Decidimos como equipo mantener `gear_any` y la semilla fija, con base en el análisis de Fabiola.
- Integramos el trabajo de los tres en el repositorio.

## Roy Silva — Preparación de datos e historial

**Herramienta:** Claude (Anthropic), usado como asistente.

**En qué partes y para qué:**
- Apoyo para crear tablas, guía en tests y comprender librerías y algoritmos con los que no estaba muy familiarizado.
- Apoyo en la redacción de cuaderno jupyter (carga, limpieza, partición, historial y pruebas de fuga), brindó otra perspectiva en la comparativa del dataset v0 vs. v1.
- Unión de los tres notebooks de trabajo en un solo cuaderno.

**Qué hice y verifiqué yo:**
- Construí la tabla base `dataset_v0` a partir de los CSV originales: carga, eliminación de las columnas posteriores a la carrera antes de unir las tablas, limpieza y variables objetivo.
- Revisé los valores outliers del dataset y decidí corregirlos (dejarlos como faltantes) en lugar de solo alertar.
- Escribí el diccionario de datos, que marca qué columnas son variables del modelo y cuáles no.
- Definí la partición cronológica (train ≤ 2003, validación 2004, test 2005).
- Genere las pruebas de partición y fuga, y revise que se cumplieran los tests correctamente para evitar cualquier tipo de fuga (`tests/`, `docs/particion_y_fuga.md`).
- Construí `dataset_v1` con el historial de caballo, jinete y entrenador, calculado solo con carreras anteriores, y reemplacé los identificadores de jinete y entrenador por sus tasas.

## Fabiola Meléndez — Explicabilidad (SHAP)

*(Por completar.)*