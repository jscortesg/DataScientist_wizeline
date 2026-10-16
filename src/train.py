"""Entrena y evalúa modelos de regresión. Ejecutar desde la raíz: python src/train.py"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data_validation import DatasetValidator
from feature_selection import PearsonCorrelationSelector
from model_search import ModelSearch
from utils import TrainingUtils


# Configuración centralizada
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TARGET = "target"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
MIN_ABS_CORRELATION = 0.05
MAX_ABS_CORRELATION = 0.95
SELECTION_TOLERANCE = 0.05


def main() -> int:
    print("=" * 68)
    print("ENTRENAMIENTO Y EVALUACIÓN DE REGRESIÓN")
    print("=" * 68)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Coloca training_data.csv en data/."
        )

    data = pd.read_csv(DATA_PATH)
    validator = DatasetValidator(
        target=TARGET,
        data_path=str(DATA_PATH),
        cv_splits=CV_SPLITS,
    )
    X, y = validator.validate(data)

    # Esta es la única separación holdout: el 20 % no entra en GridSearchCV.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    if len(X_train) < CV_SPLITS:
        raise ValueError(
            f"El 80 % de desarrollo ({len(X_train)} filas) "
            f"no permite CV de {CV_SPLITS} folds."
        )

    print(
        f"Filas totales: {len(data)} | desarrollo: {len(X_train)} | "
        f"test reservado: {len(X_test)}"
    )
    print(f"Predictores originales: {X.shape[1]} | CV: {CV_SPLITS} folds")
    print(
        "Correlación absoluta permitida: "
        f"[{MIN_ABS_CORRELATION}, {MAX_ABS_CORRELATION}]"
    )
    print("Buscando configuración con menor RMSE medio de validación cruzada...")

    parallel_jobs = TrainingUtils.get_parallel_jobs()
    available_cpus = (
        os.process_cpu_count()
        if hasattr(os, "process_cpu_count")
        else os.cpu_count()
    )
    print(f"Procesadores lógicos disponibles: {available_cpus}")
    print(f"Trabajos de entrenamiento concurrentes: {parallel_jobs}")

    search = ModelSearch.build_search(
        selector_class=PearsonCorrelationSelector,
        min_abs_correlation=MIN_ABS_CORRELATION,
        max_abs_correlation=MAX_ABS_CORRELATION,
        random_state=RANDOM_STATE,
        cv_splits=CV_SPLITS,
        n_jobs=parallel_jobs,
    )

    start_search = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed_search = time.perf_counter() - start_search

    print(
        f"Tiempo de búsqueda CV: {elapsed_search:.2f} segundos "
        f"({elapsed_search / 60:.2f} minutos)"
    )

    # Selección de candidato y entrenamiento final encapsulados en utils.py.
    best_index, best_params, cv_results = TrainingUtils.select_best_candidate(
        search=search,
        tolerance=SELECTION_TOLERANCE,
    )
    best_model = TrainingUtils.train_best_model(
        search=search,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
    )

    metrics = TrainingUtils.evaluate_model(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_results=cv_results,
        best_index=best_index,
    )
    selected_features = list(
        best_model.named_steps["selector"].selected_features_
    )
    accepted = True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar resultados detallados de validación cruzada.
    cv_results["std_cv_rmse"] = cv_results["std_test_score"]
    cv_results.sort_values(
        ["rmse_gap", "mean_cv_rmse"]
    ).to_csv(REPORTS_DIR / "cv_results.csv", index=False)

    # Reporte descriptivo calculado exclusivamente sobre el 80 % de desarrollo.
    correlations = X_train.apply(lambda col: col.corr(y_train), axis=0)
    pd.DataFrame({
        "feature": correlations.index,
        "pearson_correlation_with_target": correlations.values,
        "absolute_correlation": correlations.abs().values,
    }).sort_values(
        "absolute_correlation",
        ascending=False,
    ).to_csv(REPORTS_DIR / "feature_correlations.csv", index=False)

    (REPORTS_DIR / "selected_features.json").write_text(
        json.dumps(selected_features, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    evaluation = TrainingUtils.build_evaluation_report(
        target=TARGET,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        rows_total=len(data),
        rows_development=len(X_train),
        rows_test=len(X_test),
        cv_splits=CV_SPLITS,
        best_params=best_params,
        model=best_model,
        correlation_thresholds={
            "minimum_absolute_correlation": MIN_ABS_CORRELATION,
            "maximum_absolute_correlation": MAX_ABS_CORRELATION,
        },
        selected_features=selected_features,
        metrics=metrics,
    )
    (REPORTS_DIR / "evaluation_metrics.json").write_text(
        json.dumps(
            TrainingUtils.json_safe(evaluation),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    model_path = MODELS_DIR / "best_model.joblib"
    if accepted:
        joblib.dump(best_model, model_path)
        print(f"\nModelo aceptado y guardado: {model_path}")
    else:
        # Evita dejar un artefacto anterior con el nombre de modelo aceptado.
        if model_path.exists():
            model_path.unlink()
        print("\nCRITERIO DE ACEPTACIÓN NO SUPERADO: no se guardó best_model.joblib.")

    TrainingUtils.print_training_summary(
        model=best_model,
        metrics=metrics,
        accepted=accepted,
        selected_features=selected_features,
        reports_dir=REPORTS_DIR,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    start_total = time.perf_counter()

    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
    finally:
        elapsed_total = time.perf_counter() - start_total
        print(
            f"\nTiempo total de ejecución: {elapsed_total:.2f} segundos "
            f"({elapsed_total / 60:.2f} minutos)"
        )
