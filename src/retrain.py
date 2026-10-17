"""Evalúa estrategias de reentrenamiento sin reemplazar el modelo aprobado.

Ejecutar desde la raíz del proyecto:
python src/retrain.py --new-labeled data/new_labeled_data.csv --evaluation data/retraining_evaluation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from drift_monitoring import DataDriftMonitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "training_data.csv"
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "retraining_comparison.csv"
DEFAULT_DRIFT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "retraining_drift_psi.csv"
)

TARGET = "target"
RANDOM_STATE = 42
RECENT_WEIGHT = 2.0


class RetrainingPipeline:
    """Compara estrategias de reentrenamiento usando un modelo base aprobado."""

    def __init__(
        self,
        model_path: Path,
        reference_path: Path,
        new_labeled_path: Path,
        evaluation_path: Path,
        candidate_dir: Path,
        report_path: Path,
        drift_report_path: Path,
        target: str = TARGET,
        recent_weight: float = RECENT_WEIGHT,
        recent_fraction: float = 0.50,
    ) -> None:
        if recent_weight <= 1:
            raise ValueError("El peso de los datos recientes debe ser mayor que 1.")
        if not 0 < recent_fraction <= 1:
            raise ValueError("recent_fraction debe estar en el intervalo (0, 1].")

        self.model_path = model_path
        self.reference_path = reference_path
        self.new_labeled_path = new_labeled_path
        self.evaluation_path = evaluation_path
        self.candidate_dir = candidate_dir
        self.report_path = report_path
        self.drift_report_path = drift_report_path
        self.target = target
        self.recent_weight = recent_weight
        self.recent_fraction = recent_fraction

    def _load_labeled_data(self, path: Path) -> tuple[pd.DataFrame, pd.Series]:
        """Carga y valida un conjunto que debe contener la variable objetivo."""
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

        data = pd.read_csv(path)

        if data.empty:
            raise ValueError(f"El archivo está vacío: {path}")
        if data.columns.duplicated().any():
            raise ValueError(f"Hay columnas duplicadas en {path}.")
        if self.target not in data.columns:
            raise ValueError(
                f"El archivo {path} debe contener la columna '{self.target}'."
            )

        y = pd.to_numeric(data[self.target], errors="raise").astype(float)
        X = data.drop(columns=[self.target]).copy()

        if X.empty:
            raise ValueError(f"No hay características predictoras en {path}.")

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Hay características no numéricas en {path}: {non_numeric}"
            )

        X = X.astype(float)
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                f"Hay valores faltantes o no finitos en los predictores de {path}."
            )
        if y.isna().any() or not np.isfinite(y.to_numpy()).all():
            raise ValueError(
                f"'{self.target}' contiene valores faltantes o no finitos en {path}."
            )

        return X, y

    @staticmethod
    def _align_features(
        X: pd.DataFrame,
        expected_features: list[str],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Comprueba las características y ajusta su orden al conjunto de referencia."""
        missing = sorted(set(expected_features) - set(X.columns))
        unexpected = sorted(set(X.columns) - set(expected_features))
        if missing or unexpected:
            raise ValueError(
                f"Las características de {dataset_name} no coinciden. "
                f"Faltantes: {missing}; adicionales: {unexpected}."
            )
        return X.loc[:, expected_features]

    @staticmethod
    def _metrics(
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Calcula métricas de regresión sobre el conjunto de evaluación."""
        predictions = np.asarray(model.predict(X)).reshape(-1)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions)),
            "mean_error": float(np.mean(y.to_numpy() - predictions)),
        }

    def _fit_candidate(
        self,
        name: str,
        base_model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[Pipeline, dict[str, float]]:
        """Ajusta una copia del pipeline aprobado y mide su rendimiento."""
        candidate = clone(base_model)

        # La ponderación se aplica al estimador final, no al selector de variables.
        if sample_weights is None:
            candidate.fit(X, y)
        else:
            candidate.fit(X, y, model__sample_weight=sample_weights)

        return candidate, self._metrics(candidate, self.X_eval, self.y_eval)

    def run(self) -> pd.DataFrame:
        """Compara estrategias y guarda candidatos sin promoverlos automáticamente."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo aprobado: {self.model_path}"
            )

        base_model: Any = joblib.load(self.model_path)
        if not isinstance(base_model, Pipeline):
            raise TypeError(
                "El modelo aprobado no es un sklearn Pipeline compatible "
                "con el flujo de reentrenamiento."
            )

        X_old, y_old = self._load_labeled_data(self.reference_path)
        X_new, y_new = self._load_labeled_data(self.new_labeled_path)
        self.X_eval, self.y_eval = self._load_labeled_data(self.evaluation_path)

        # Analizar cambios de distribución antes del reentrenamiento.
        print("\nAnalizando drift entre los datos históricos y los nuevos...")

        drift_monitor = DataDriftMonitor(
            reference_path=self.reference_path,
            current_path=self.new_labeled_path,
            report_path=self.drift_report_path,
        )

        drift_report = drift_monitor.generate_report()

        print("\nInforme de drift:")
        print(drift_report.to_string(index=False))
        print(f"\nInforme de drift guardado en: {self.drift_report_path}")

        if len(X_new) < 2:
            raise ValueError(
                "Se necesitan al menos dos filas nuevas etiquetadas."
            )

        expected_features = list(X_old.columns)
        X_new = self._align_features(X_new, expected_features, "los datos nuevos")
        self.X_eval = self._align_features(
            self.X_eval, expected_features, "el conjunto de evaluación"
        )

        if self.X_eval.empty:
            raise ValueError("El conjunto de evaluación no puede estar vacío.")

        # El conjunto de evaluación debe ser independiente de los datos usados
        # para entrenar; el script no puede verificar la ausencia de solapamiento.
        X_combined = pd.concat([X_old, X_new], ignore_index=True)
        y_combined = pd.concat([y_old, y_new], ignore_index=True)

        recent_rows = max(2, int(np.ceil(len(X_new) * self.recent_fraction)))
        recent_rows = min(recent_rows, len(X_new))
        X_recent = X_new.iloc[-recent_rows:].reset_index(drop=True)
        y_recent = y_new.iloc[-recent_rows:].reset_index(drop=True)

        strategies: list[tuple[str, pd.DataFrame, pd.Series, np.ndarray | None]] = [
            ("historico_mas_nuevos", X_combined, y_combined, None),
            ("solo_nuevos_recientes", X_recent, y_recent, None),
        ]

        weights = np.concatenate(
            [
                np.ones(len(X_old), dtype=float),
                np.full(len(X_new), self.recent_weight, dtype=float),
            ]
        )
        strategies.append(
            ("historico_ponderado", X_combined, y_combined, weights)
        )

        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, str | float]] = []

        baseline_metrics = self._metrics(
            base_model, self.X_eval, self.y_eval
        )
        results.append(
            {
                "strategy": "modelo_vigente",
                **baseline_metrics,
                "candidate_path": str(self.model_path),
            }
        )

        for name, X_train, y_train, sample_weights in strategies:
            print(f"\nEntrenando estrategia: {name}")
            candidate, metrics = self._fit_candidate(
                name,
                base_model,
                X_train,
                y_train,
                sample_weights,
            )

            candidate_path = self.candidate_dir / f"{name}.joblib"
            joblib.dump(candidate, candidate_path)

            results.append(
                {
                    "strategy": name,
                    **metrics,
                    "candidate_path": str(candidate_path),
                }
            )
            print(
                f"RMSE: {metrics['rmse']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"R²: {metrics['r2']:.4f}"
            )

        report = pd.DataFrame(results).sort_values("rmse")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(self.report_path, index=False)

        metadata_path = self.report_path.with_suffix(".json")
        metadata = {
            "evaluation_path": str(self.evaluation_path),
            "target": self.target,
            "recent_weight": self.recent_weight,
            "recent_fraction": self.recent_fraction,
            "automatic_promotion": False,
            "note": (
                "Los candidatos se guardan por separado. La menor RMSE "
                "no implica aprobación automática; revise métricas, "
                "representatividad y criterios de aceptación."
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\nComparación de estrategias (ordenada por RMSE):")
        print(report.to_string(index=False))
        print(f"\nInforme CSV: {self.report_path}")
        print(f"Metadatos: {metadata_path}")
        print(
            "\nLos candidatos NO reemplazan automáticamente el modelo aprobado."
        )

        return report


def parse_args() -> argparse.Namespace:
    """Define los argumentos de ejecución del script."""
    parser = argparse.ArgumentParser(
        description="Compara estrategias de reentrenamiento con datos etiquetados."
    )
    parser.add_argument(
        "--new-labeled",
        type=Path,
        required=True,
        help="CSV de datos recientes que contiene las características y target.",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        required=True,
        help="CSV independiente y etiquetado para evaluar todos los modelos.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="CSV histórico de entrenamiento.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Modelo aprobado que se usará como base de configuración.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="Directorio donde se guardan los candidatos.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Ruta del informe comparativo CSV.",
    )
    parser.add_argument(
        "--recent-weight",
        type=float,
        default=RECENT_WEIGHT,
        help="Peso asignado a cada fila nueva en la estrategia ponderada.",
    )
    parser.add_argument(
        "--recent-fraction",
        type=float,
        default=0.50,
        help="Fracción final de las filas nuevas usada en la ventana reciente.",
    )
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=DEFAULT_DRIFT_REPORT_PATH,
        help="Ruta del informe PSI de los datos nuevos.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada del script."""
    args = parse_args()
    pipeline = RetrainingPipeline(
        model_path=args.model,
        reference_path=args.reference,
        new_labeled_path=args.new_labeled,
        evaluation_path=args.evaluation,
        candidate_dir=args.candidate_dir,
        report_path=args.report,
        drift_report_path=args.drift_report,
        recent_weight=args.recent_weight,
        recent_fraction=args.recent_fraction,
    )
    pipeline.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
