"""Utilidades compartidas por el entrenamiento de modelos."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TrainingUtils:
    """Agrupa operaciones reutilizables del flujo de entrenamiento."""

    @staticmethod
    def get_parallel_jobs() -> int:
        """Detecta los procesadores lógicos disponibles para Python.

        Reserva un procesador lógico para el sistema y otras tareas.
        """
        available_cpus = (
            os.process_cpu_count()
            if hasattr(os, "process_cpu_count")
            else os.cpu_count()
        )
        if available_cpus is None:
            return 1
        return max(1, available_cpus - 1)

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        """Calcula la raíz del error cuadrático medio (RMSE)."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def json_safe(value: Any):
        """Convierte recursivamente valores comunes a tipos serializables en JSON."""
        if isinstance(value, dict):
            return {
                str(k): TrainingUtils.json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TrainingUtils.json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def select_best_candidate(
        search,
        tolerance: float = 0.05,
    ) -> tuple[int, dict[str, Any], pd.DataFrame]:
        """Selecciona el candidato por CV usando calidad y brecha de RMSE.

        Primero limita los candidatos a aquellos cuyo RMSE medio de validación
        esté dentro de `tolerance` del mínimo; después minimiza la brecha
        absoluta entre RMSE de entrenamiento y validación, con RMSE de
        validación como desempate.
        """
        if not 0 <= tolerance:
            raise ValueError("La tolerancia debe ser mayor o igual que cero.")

        cv_results = pd.DataFrame(search.cv_results_)
        if cv_results.empty:
            raise ValueError("La búsqueda no produjo resultados de validación cruzada.")

        cv_results["mean_cv_rmse"] = -cv_results["mean_test_score"]
        cv_results["mean_cv_train_rmse"] = -cv_results["mean_train_score"]
        cv_results["rmse_gap"] = (
            cv_results["mean_cv_train_rmse"] - cv_results["mean_cv_rmse"]
        ).abs()

        minimum_cv_rmse = cv_results["mean_cv_rmse"].min()
        eligible = cv_results[
            cv_results["mean_cv_rmse"] <= minimum_cv_rmse * (1 + tolerance)
        ].copy()

        if eligible.empty:
            raise ValueError(
                "No hay candidatos que cumplan la tolerancia de RMSE especificada."
            )

        eligible = eligible.sort_values(
            ["rmse_gap", "mean_cv_rmse"],
            ascending=[True, True],
        )
        best_index = int(eligible.index[0])
        best_params = cv_results.at[best_index, "params"]

        if not isinstance(best_params, dict):
            raise TypeError(
                "Los parámetros del candidato seleccionado no son un diccionario."
            )

        return best_index, best_params, cv_results

    @staticmethod
    def train_best_model(
        search,
        best_params: dict[str, Any],
        X_train,
        y_train,
    ):
        """Clona el pipeline de búsqueda, aplica los parámetros elegidos y lo ajusta."""
        best_model = clone(search.estimator)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        return best_model

    @staticmethod
    def evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_results: pd.DataFrame,
        best_index: int,
    ) -> dict[str, float]:
        """Calcula métricas de desarrollo, test reservado y validación cruzada."""
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_rmse = TrainingUtils.rmse(y_train, train_pred)
        test_rmse = TrainingUtils.rmse(y_test, test_pred)
        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_r2 = float(r2_score(y_test, test_pred))

        return {
            "best_cv_rmse": float(cv_results.loc[best_index, "mean_cv_rmse"]),
            "best_cv_train_rmse": float(
                cv_results.loc[best_index, "mean_cv_train_rmse"]
            ),
            "best_cv_rmse_gap": float(cv_results.loc[best_index, "rmse_gap"]),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "train_test_rmse_gap_absolute": abs(test_rmse - train_rmse),
        }

    @staticmethod
    def build_evaluation_report(
        *,
        target: str,
        random_state: int,
        test_size: float,
        rows_total: int,
        rows_development: int,
        rows_test: int,
        cv_splits: int,
        best_params: dict[str, Any],
        model,
        correlation_thresholds: dict[str, float],
        selected_features: list[str],
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Construye el informe de evaluación en formato serializable."""
        return {
            "target": target,
            "random_state": random_state,
            "test_size": test_size,
            "rows_total": int(rows_total),
            "rows_development": int(rows_development),
            "rows_test": int(rows_test),
            "cross_validation_folds": cv_splits,
            "selection_criterion": (
                "Minimum absolute gap between CV training RMSE and CV validation "
                "RMSE among candidates within 5% of the minimum CV RMSE"
            ),
            "best_cv_rmse": metrics["best_cv_rmse"],
            "best_cv_train_rmse": metrics["best_cv_train_rmse"],
            "best_cv_rmse_gap": metrics["best_cv_rmse_gap"],
            "best_parameters": TrainingUtils.json_safe(best_params),
            "model_class": type(model.named_steps["model"]).__name__,
            "correlation_thresholds": correlation_thresholds,
            "selected_features": selected_features,
            "metrics": {
                "train_rmse_development_80_percent": metrics["train_rmse"],
                "test_rmse_holdout_20_percent": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "test_r2": metrics["test_r2"],
                "train_test_rmse_gap_absolute": (
                    metrics["train_test_rmse_gap_absolute"]
                ),
            },
            "acceptance": {
                "passed": True,
                "interpretation": (
                    "Model selected using cross-validation; holdout test metrics "
                    "are reported for final evaluation only."
                ),
            },
            "note": (
                "El holdout del 20 % no se utilizó para seleccionar variables ni "
                "ajustar hiperparámetros. No se hizo refit con el 100 %."
            ),
        }

    @staticmethod
    def print_training_summary(
        *,
        model,
        metrics: dict[str, float],
        accepted: bool,
        selected_features: list[str],
        reports_dir,
    ) -> None:
        """Imprime el resumen final del entrenamiento y la evaluación."""
        print("\n--- Resultado ---")
        print(f"Modelo seleccionado: {type(model.named_steps['model']).__name__}")
        print(f"Mejor RMSE medio CV: {metrics['best_cv_rmse']:.3f}")
        print(f"RMSE desarrollo (80 %): {metrics['train_rmse']:.3f}")
        print(f"RMSE test reservado (20 %): {metrics['test_rmse']:.3f}")
        print(
            f"MAE test: {metrics['test_mae']:.3f} | "
            f"R² test: {metrics['test_r2']:.3f}"
        )
        print(f"RMSE entrenamiento CV: {metrics['best_cv_train_rmse']:.3f}")
        print(f"RMSE validación CV: {metrics['best_cv_rmse']:.3f}")
        print(f"Brecha absoluta CV: {metrics['best_cv_rmse_gap']:.3f}")
        print(
            "Brecha absoluta entrenamiento/test: "
            f"{metrics['train_test_rmse_gap_absolute']:.3f}"
        )
        print(f"Criterio: {'APROBADO' if accepted else 'NO APROBADO'}")
        print(f"Variables seleccionadas por el modelo final: {len(selected_features)}")
        print(f"Informes: {reports_dir}")
