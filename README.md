# Data Scientist Regression Project

**Language / Idioma:** [Español](#español) | [English](#english)

---

## Español

### 1. Descripción

Este proyecto entrena y evalúa modelos de regresión, selecciona una configuración mediante validación cruzada y genera predicciones para un conjunto de prueba ciego.

El flujo se ejecuta desde un cuaderno Jupyter que orquesta los scripts de entrenamiento y predicción. La lógica permanece separada en módulos Python para facilitar su mantenimiento y reutilización.

### 2. Requisitos

- Python 3.13.9.
- Git, si se desea clonar el repositorio.
- Las dependencias indicadas en `requirements.txt`.

### 3. Estructura del proyecto

```text
DataScientist_wizeline/
├── data/
│   ├── training_data.csv
│   └── blind_test_data.csv
├── models/                         # Generado localmente; ignorado por Git
├── outputs/                        # Generado localmente; ignorado por Git
├── reports/                        # Informes del entrenamiento
├── src/
│   ├── data_validation.py          # Validación de datos
│   ├── feature_selection.py        # Selector de correlación de Pearson
│   ├── model_search.py             # Pipeline y búsqueda de modelos
│   ├── train.py                    # Entrenamiento y evaluación
│   ├── predict.py                  # Predicción sobre datos ciegos
│   └── utils.py                    # Utilidades del entrenamiento
├── orquestador_entrenamiento_prediccion.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

Las carpetas `models/` y `outputs/` se crean cuando se ejecutan los scripts si no existen. Sus contenidos están excluidos del control de versiones mediante `.gitignore`.

### 4. Preparación del entorno

Desde la raíz del proyecto, crea y activa un entorno virtual.

**Windows / PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno por la política de ejecución, consulta la configuración de tu equipo o activa el entorno desde la terminal integrada de VS Code según las políticas de tu organización.

### 5. Preparar los datos

Coloca los archivos de entrada en `data/`:

- `training_data.csv`: datos de entrenamiento. Debe incluir la variable objetivo `target` y predictores numéricos.
- `blind_test_data.csv`: conjunto ciego para generar predicciones. Debe incluir las variables predictoras esperadas por el modelo.

El validador rechaza columnas predictoras no numéricas y valores faltantes o infinitos; no realiza imputación automática. Las variables del conjunto ciego deben coincidir con las que espera el modelo.

### 6. Ejecutar el proyecto desde el cuaderno

1. Abre `orquestador_entrenamiento_prediccion.ipynb` en VS Code o Jupyter.
2. Selecciona como kernel el intérprete de `.venv`.
3. Ejecuta las celdas en orden.

El cuaderno ejecuta primero `src/train.py` y, solo si el entrenamiento termina correctamente, ejecuta `src/predict.py`. Los mensajes de consola de ambos scripts se muestran en el cuaderno.

### 7. Ejecutar los scripts manualmente

También puedes ejecutar cada etapa desde la raíz del proyecto:

```powershell
python -m py_compile src/data_validation.py src/feature_selection.py src/model_search.py src/utils.py src/train.py src/predict.py
python -u src/train.py
python -u src/predict.py
```

La predicción requiere que el modelo haya sido generado previamente por el entrenamiento.

### 8. Flujo de entrenamiento

El script de entrenamiento:

1. Lee y valida `data/training_data.csv`.
2. Separa los datos en desarrollo (80 %) y test reservado (20 %).
3. Ejecuta una búsqueda de hiperparámetros con validación cruzada de 5 folds en el conjunto de desarrollo.
4. Aplica la selección de variables por correlación de Pearson entre la variable objetivo y cada atributo, teniendo en cuenta umbrales mínimo (0.5%) y máximo (0.95%) en valor absoluto de estos valores.
5. Entre candidatos cuyo RMSE medio de validación cruzada esté dentro del 5 % del mínimo asociado a esta métrica, se selecciona el de menor brecha absoluta entre RMSE de entrenamiento y validación; el RMSE de validación se usa como desempate.
6. Entrena la configuración seleccionada con el conjunto de desarrollo.
7. Calcula las métricas del conjunto de desarrollo y del test reservado.
8. Guarda el modelo y los informes en reports.

El test reservado se usa para evaluación final, no para seleccionar variables ni hiperparámetros.

### 9. Archivos generados

Después del entrenamiento:

- `models/best_model.joblib`: pipeline entrenado completo, incluido el selector de variables.
- `reports/cv_results.csv`: resultados de validación cruzada.
- `reports/feature_correlations.csv`: correlaciones de los predictores con la variable objetivo calculadas sobre el conjunto de desarrollo.
- `reports/selected_features.json`: variables seleccionadas.
- `reports/evaluation_metrics.json`: configuración y métricas de evaluación.

Después de la predicción:

- `outputs/blind_test_predictions.csv`: filas originales de `blind_test_data.csv` con la columna adicional `target_pred`.

`predict.py` carga el pipeline guardado y aplica la selección de variables aprendida durante el entrenamiento. No vuelve a calcular las correlaciones ni ajusta el modelo con el conjunto ciego.

### 10. Métricas y limitaciones

Las métricas del test reservado describen el rendimiento observado en la partición de evaluación disponible. Las predicciones del conjunto ciego no pueden evaluarse sin conocer sus valores reales de `target`.

El estado `APROBADO` impreso por el script representa el criterio de aceptación implementado en el proyecto; no garantiza el rendimiento sobre todos los datos futuros.

### 11. Reproducibilidad y control de versiones

- La configuración incluye una semilla aleatoria fija para la partición y la validación cruzada.
- `models/` y `outputs/` están ignoradas por Git porque son artefactos generados.
- `reports/` no está excluida por las reglas indicadas aquí; revisa `.gitignore` y la política de datos antes de versionar los informes.
- Para reproducir los resultados, usa los mismos datos y las mismas versiones de dependencias.


### 12. Estrategia de monitorización, deriva de datos y reentrenamiento

La solución actual implementa un flujo reproducible de entrenamiento por lotes y generación de predicciones. Como primer paso hacia un entorno de producción, se propone la siguiente estrategia de monitorización y reentrenamiento.

### 13. Monitorización de la calidad y la deriva de los datos

Los nuevos lotes de datos deben validarse antes de generar predicciones. La validación incluye comprobar el esquema de características esperado, los tipos de datos numéricos, los valores faltantes y los valores no finitos.

La deriva de datos (*data drift*) puede monitorizarse comparando las distribuciones de las 20 variables de entrada con las de un conjunto de referencia. Las medidas estadísticas y basadas en distribuciones, como el Índice de Estabilidad de la Población (*Population Stability Index*, PSI), permiten identificar cambios que requieren investigación. Los umbrales iniciales deben considerarse provisionales y calibrarse utilizando datos representativos.

La detección de deriva no implica necesariamente una degradación del modelo. Los cambios en la distribución de las predicciones también se monitorizan como señales de alerta, pero no constituyen una medida directa de la precisión del modelo.

#### Monitorización del rendimiento del modelo

Cuando estén disponibles los valores reales de la variable objetivo, las predicciones deben asociarse con sus correspondientes valores reales para calcular métricas de regresión como RMSE, MAE y R².

Si los valores reales de la variable objetivo no están disponibles, no es posible medir directamente los errores reales de las predicciones. Aun así, se pueden monitorizar la deriva de las variables de entrada y los cambios en la distribución de las predicciones, aunque estos indicadores no permiten confirmar una pérdida de precisión del modelo.

#### Reentrenamiento y aprobación de modelos

Se puede considerar el reentrenamiento cuando se observe una degradación del rendimiento en datos etiquetados suficientemente representativos, o cuando se produzcan cambios relevantes en los datos que justifiquen una nueva evaluación.

Un modelo candidato debe evaluarse frente al modelo actualmente aprobado utilizando un conjunto de validación independiente y criterios de aceptación definidos previamente. Únicamente deben incorporarse los modelos candidatos que cumplan estos criterios. Se deben conservar las versiones de los modelos, los resultados de las evaluaciones y la posibilidad de volver a la versión anterior.

#### Mejoras futuras

Como trabajo futuro, se contempla la generación automatizada de informes de deriva, la monitorización programada, el seguimiento de experimentos y la gestión de versiones de modelos mediante MLflow, así como la implementación de un flujo controlado de reentrenamiento.

En la implementación actual no se habilita el reentrenamiento automático basado exclusivamente en la detección de deriva de datos.

### 14. Monitorización de deriva y evaluación de estrategias de reentrenamiento

#### Monitorización de deriva de datos (Data Drift)

Se incorporó `src/drift_monitoring.py`, que implementa la clase `DataDriftMonitor` para comparar las distribuciones de las variables predictoras entre un conjunto de referencia y uno reciente mediante el índice **Population Stability Index (PSI)**.

El monitor:

- Carga los conjuntos de datos de referencia y recientes.
- Excluye la columna objetivo `target` de ambos conjuntos, cuando está presente, para analizar únicamente las variables predictoras.
- Valida que ambos conjuntos tengan observaciones, que no existan columnas duplicadas y que coincida el esquema de características.
- Comprueba que las variables sean numéricas y que no contengan valores faltantes o no finitos.
- Define los intervalos del PSI a partir de los cuantiles del conjunto de referencia.
- Clasifica cada variable de acuerdo con los umbrales configurados y guarda el resultado en un CSV.

Los umbrales predeterminados son:

| PSI | Clasificación |
|---|---|
| PSI < 0.20 | `SIN_ALERTA` |
| 0.20 ≤ PSI < 0.30 | `INVESTIGAR` |
| PSI ≥ 0.30 | `ALERTA` |

Estos umbrales son valores de referencia configurables, no reglas universales. El PSI detecta cambios en la distribución de las variables de entrada; por sí solo no demuestra una degradación del modelo ni mide el error predictivo.

#### Integración con el reentrenamiento

Se amplió `src/retrain.py` para ejecutar el análisis de deriva antes de entrenar y comparar las estrategias candidatas. La comparación del drift usa:

- **Referencia:** `data/training_data.csv`.
- **Datos recientes etiquetados:** `data/new_labeled_data.csv`.
- **Informe de drift:** `reports/retraining_drift_psi.csv`.

El resultado de drift es diagnóstico: no bloquea el entrenamiento, no selecciona automáticamente una estrategia y no reemplaza el modelo vigente.

#### Estrategias de reentrenamiento evaluadas

El script compara el modelo vigente con tres estrategias:

1. **`historico_mas_nuevos`:** entrena con los datos históricos y los nuevos.
2. **`solo_nuevos_recientes`:** entrena con la fracción final de los nuevos registros, asumiendo que las filas están ordenadas cronológicamente.
3. **`historico_ponderado`:** combina los datos históricos y nuevos, asignando mayor peso a los registros recientes.

La evaluación se realiza con `data/retraining_evaluation.csv`, un conjunto etiquetado separado que no se utiliza para ajustar los candidatos. Se calculan RMSE, MAE, R² y error medio. Para que esta comparación tenga valor real, el conjunto de evaluación debe ser independiente y contar con etiquetas reales representativas del uso esperado.

Los candidatos se guardan por separado en `models/candidates/`. El script genera `reports/retraining_comparison.csv` y `reports/retraining_comparison.json`. **El modelo aprobado no se reemplaza automáticamente**; la promoción requiere una revisión posterior de las métricas y de los criterios de aceptación.

#### Integración en el cuaderno

El cuaderno `orquestador_entrenamiento_prediccion.ipynb` se amplió con una sección para ejecutar el flujo de prueba de drift y reentrenamiento con los datos sintéticos:

1. Comprueba que existan `src/retrain.py` y los dos CSV sintéticos.
2. Ejecuta el script de reentrenamiento con los argumentos correspondientes.
3. Comprueba la generación de los informes de drift y comparación.
4. Carga y muestra ambos informes dentro del cuaderno.

La función auxiliar `run_script()` admite argumentos adicionales, conservando su uso previo en las etapas existentes. La nueva sección es adicional al flujo principal de entrenamiento, monitorización del conjunto ciego y generación de predicciones.

#### Prueba técnica con datos sintéticos

Se probó el flujo con `data/new_labeled_data.csv` y `data/retraining_evaluation.csv`. En la ejecución de prueba, el PSI señaló `feature_5` (aprox. 0.228) y `feature_14` (aprox. 0.201) como `INVESTIGAR`; las otras 18 variables quedaron por debajo de 0.20. No hubo variables con estado `ALERTA`.

En la comparación sobre el conjunto sintético de evaluación, el modelo vigente obtuvo un RMSE aproximado de 1.740; los candidatos obtuvieron RMSE aproximados de 2.072 (`historico_mas_nuevos`), 2.092 (`historico_ponderado`) y 3.329 (`solo_nuevos_recientes`). Estos valores documentan una ejecución técnica concreta y pueden variar si cambian los datos o la configuración.

**Limitación importante:** los dos CSV usados en esta prueba contienen datos sintéticos. Los resultados comprueban que el flujo se ejecuta y produce informes, pero no demuestran que exista drift real ni permiten concluir qué estrategia funcionará mejor en producción.

#### Archivos e informes generados

| Archivo | Propósito |
|---|---|
| `reports/data_drift_psi.csv` | Informe PSI del monitor de drift utilizado en el flujo principal. |
| `reports/retraining_drift_psi.csv` | Informe PSI entre el histórico y los nuevos datos etiquetados. |
| `reports/retraining_comparison.csv` | Métricas del modelo vigente y de los candidatos. |
| `reports/retraining_comparison.json` | Metadatos y advertencia sobre la promoción manual. |
| `models/candidates/` | Modelos candidatos guardados sin reemplazar el modelo aprobado. |


---

## English

### 1. Overview

This project trains and evaluates regression models, selects a configuration using cross-validation, and generates predictions for a blind test dataset.

A Jupyter notebook orchestrates the training and prediction scripts. The implementation remains split into Python modules to support maintenance and reuse.

### 2. Requirements

- Python 3.13.9.
- Git, if you want to clone the repository.
- The dependencies listed in `requirements.txt`.

### 3. Project structure

```text
DataScientist_wizeline/
├── data/
│   ├── training_data.csv
│   └── blind_test_data.csv
├── models/                         # Generated locally; ignored by Git
├── outputs/                        # Generated locally; ignored by Git
├── reports/                        # Training reports
├── src/
│   ├── data_validation.py          # Data validation
│   ├── feature_selection.py        # Pearson correlation selector
│   ├── model_search.py             # Model pipeline and search
│   ├── train.py                    # Training and evaluation
│   ├── predict.py                  # Blind-test prediction
│   └── utils.py                    # Training utilities
├── orquestador_entrenamiento_prediccion.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` and `outputs/` directories are created by the scripts when needed. Their contents are excluded from version control through `.gitignore`.

### 4. Environment setup

From the project root, create and activate a virtual environment.

**Windows / PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, follow your machine's execution-policy guidance or select the environment through VS Code's integrated terminal and kernel settings, according to your organization's policies.

### 5. Prepare the data

Place the input files in `data/`:

- `training_data.csv`: training data. It must include the target column `target` and numeric predictors.
- `blind_test_data.csv`: blind dataset used to generate predictions. It must contain the predictor columns expected by the trained model.

The validator rejects non-numeric predictors and missing or infinite values; it does not automatically impute them. The blind dataset's predictor columns must match the columns expected by the model.

### 6. Run the project from the notebook

1. Open `orquestador_entrenamiento_prediccion.ipynb` in VS Code or Jupyter.
2. Select the `.venv` interpreter as the notebook kernel.
3. Run the cells from top to bottom.

The notebook runs `src/train.py` first. It runs `src/predict.py` only if training completes successfully. Console messages from both scripts are displayed in the notebook.

### 7. Run the scripts manually

You can also run each stage from the project root:

```powershell
python -m py_compile src/data_validation.py src/feature_selection.py src/model_search.py src/utils.py src/train.py src/predict.py
python -u src/train.py
python -u src/predict.py
```

Prediction requires a model generated by the training stage.

### 8. Training workflow

The training script:

1. Reads and validates `data/training_data.csv`.
2. Splits the data into a development set (80%) and a reserved test set (20%).
3. Runs a hyperparameter search using 5-fold cross-validation on the development set.
4. Applies Pearson-correlation feature selection between the target variable and each one of the attributes; such selection considers both a minimum (0.5%) and a maximum (0.95%) among those absolute correlation values.
5. Among candidates whose mean cross-validation RMSE is within 5% of the minimum, selects the candidate with the smallest absolute gap between training and validation RMSE; validation RMSE breaks ties.
6. Fits the selected configuration on the development set.
7. Computes development and reserved-test metrics.
8. Saves the model and reports.

The reserved test set is used for final evaluation, not for feature selection or hyperparameter tuning.

### 9. Generated files

After training:

- `models/best_model.joblib`: the complete trained pipeline, including feature selection.
- `reports/cv_results.csv`: cross-validation results.
- `reports/feature_correlations.csv`: predictor-to-target correlations computed on the development set.
- `reports/selected_features.json`: selected features.
- `reports/evaluation_metrics.json`: evaluation configuration and metrics.

After prediction:

- `outputs/blind_test_predictions.csv`: the original rows from `blind_test_data.csv` plus the `target_pred` column.

`predict.py` loads the saved pipeline and applies the feature selection learned during training. It does not recompute correlations or fit the model on the blind dataset.

### 10. Metrics and limitations

Reserved-test metrics describe performance on the available evaluation split. Blind-test predictions cannot be evaluated without the true `target` values.

The `APROBADO` status printed by the script represents the acceptance criterion implemented in this project; it is not a guarantee of performance on all future data.

### 11. Reproducibility and version control

- The configuration uses a fixed random seed for the split and cross-validation.
- `models/` and `outputs/` are ignored by Git because they contain generated artifacts.
- `reports/` is not excluded by the rules described here; review `.gitignore` and your data policy before versioning reports.
- For reproducibility, use the same data and dependency versions.

### 12. Monitoring, Data Drift and Retraining Strategy

The current solution implements a reproducible batch-training and prediction workflow. As a first step toward production readiness, the following monitoring and retraining strategy is proposed.

#### Data quality and drift monitoring

Incoming prediction batches must be validated before inference. Validation includes checking the expected feature schema, numeric data types, missing values, and non-finite values.

Data drift can be monitored by comparing the distributions of the 20 input features against a reference dataset. Statistical and distribution-based measures, such as the Population Stability Index (PSI), can help identify changes that require investigation. Initial thresholds must be treated as provisional and calibrated using representative data.

A detected drift does not necessarily imply model degradation. Prediction distribution changes are also monitored as warning signals, not as direct measures of prediction accuracy.

#### Model performance monitoring

When ground-truth target values become available, predictions must be matched with their corresponding actual values to calculate regression metrics such as RMSE, MAE, and R².

If target values are unavailable, actual prediction errors cannot be measured directly. Input and prediction drift can still be monitored, but these indicators cannot confirm a loss of model accuracy.

#### Retraining and model promotion

Retraining may be considered when performance degradation is observed on sufficiently representative labeled data, or when meaningful changes in the data justify a new evaluation.

A candidate model must be evaluated against the currently approved model using an independent validation dataset and predefined acceptance criteria. Only candidates that satisfy these criteria should be promoted. Model versions, evaluation results, and the ability to revert to the previous version should be preserved.

#### Future improvements

Future work may include automated drift reports, scheduled monitoring, experiment tracking and model version management with MLflow, and a controlled retraining pipeline. Automatic retraining based solely on drift detection is not enabled in the current implementation.


### 14. Data drift monitoring

`src/drift_monitoring.py` introduces the `DataDriftMonitor` class to compare predictor distributions between a reference dataset and a recent dataset using the **Population Stability Index (PSI)**.

The monitor:

- Loads the reference and recent datasets.
- Excludes the `target` column from both datasets when present, so the analysis focuses on predictors.
- Validates that both datasets contain observations, have no duplicate columns, and share the same feature schema.
- Checks that features are numeric and contain no missing or non-finite values.
- Builds PSI bins from the reference dataset's quantiles.
- Assigns a status to each feature using configurable thresholds and writes the results to a CSV report.

Default thresholds:

| PSI | Status |
|---|---|
| PSI < 0.20 | `SIN_ALERTA` (no alert) |
| 0.20 ≤ PSI < 0.30 | `INVESTIGAR` (investigate) |
| PSI ≥ 0.30 | `ALERTA` (alert) |

These are configurable reference thresholds, not universal rules. PSI detects changes in input-feature distributions; by itself, it does not establish model degradation or measure predictive error.

#### Integration with retraining

`src/retrain.py` was extended to run drift analysis before training and comparing candidate strategies. The drift comparison uses:

- **Reference dataset:** `data/training_data.csv`.
- **Recent labeled data:** `data/new_labeled_data.csv`.
- **Drift report:** `reports/retraining_drift_psi.csv`.

The drift result is diagnostic only: it does not block training, automatically choose a strategy, or replace the current approved model.

#### Retraining strategies

The script compares the current model with three strategies:

1. **`historico_mas_nuevos`:** train on historical and new labeled data.
2. **`solo_nuevos_recientes`:** train on the final fraction of new records, assuming rows are chronologically ordered.
3. **`historico_ponderado`:** combine historical and new data while assigning greater weight to recent records.

Models are evaluated on `data/retraining_evaluation.csv`, a separate labeled dataset that is not used to fit the candidates. The script calculates RMSE, MAE, R², and mean error. For a meaningful real-world comparison, the evaluation set must be independent and contain genuine labels representative of the intended use case.

Candidate models are saved separately under `models/candidates/`. The script generates `reports/retraining_comparison.csv` and `reports/retraining_comparison.json`. **The approved model is not replaced automatically**; promotion requires a subsequent review of the metrics and acceptance criteria.

#### Notebook integration

The notebook `orquestador_entrenamiento_prediccion.ipynb` was extended with a section that runs the drift and retraining test using synthetic data:

1. Checks that `src/retrain.py` and both synthetic CSV files exist.
2. Runs the retraining script with the required arguments.
3. Checks that the drift and model-comparison reports were created.
4. Loads and displays both reports in the notebook.

The helper function `run_script()` now accepts additional command-line arguments while preserving its previous use in existing stages. This new section supplements the main training, blind-test drift monitoring, and prediction workflow.

#### Synthetic-data smoke test

The workflow was tested with `data/new_labeled_data.csv` and `data/retraining_evaluation.csv`. In the recorded test run, PSI marked `feature_5` (approximately 0.228) and `feature_14` (approximately 0.201) as `INVESTIGAR` (investigate); the remaining 18 features were below 0.20. No feature reached the `ALERTA` threshold.

On the synthetic evaluation set, the current model achieved an RMSE of approximately 1.740. Candidate RMSE values were approximately 2.072 (`historico_mas_nuevos`), 2.092 (`historico_ponderado`), and 3.329 (`solo_nuevos_recientes`). These values document one technical test run and may change with different data or settings.

**Important limitation:** both CSV files used in this test contain synthetic data. The results confirm that the workflow runs and generates reports, but they do not demonstrate real data drift or establish which strategy would perform best in production.

#### Generated files and reports

| File | Purpose |
|---|---|
| `reports/data_drift_psi.csv` | PSI report from the drift monitor used in the main workflow. |
| `reports/retraining_drift_psi.csv` | PSI report comparing historical and new labeled data. |
| `reports/retraining_comparison.csv` | Metrics for the current model and candidate models. |
| `reports/retraining_comparison.json` | Metadata and reminder that model promotion is manual. |
| `models/candidates/` | Candidate models saved without replacing the approved model. |
