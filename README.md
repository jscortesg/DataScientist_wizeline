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
