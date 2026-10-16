
"""Generación de predicciones sobre el conjunto de prueba ciego."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
INPUT_PATH = PROJECT_ROOT / "data" / "blind_test_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "blind_test_predictions.csv"

TARGET = "target"
PREDICTION_COLUMN = "target_pred"


class PredictionPipeline:
    """Carga el modelo entrenado y genera predicciones para datos nuevos."""

    def __init__(
        self,
        model_path: Path,
        input_path: Path,
        output_path: Path,
        target: str = TARGET,
        prediction_column: str = PREDICTION_COLUMN,
    ) -> None:
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.target = target
        self.prediction_column = prediction_column

    def run(self) -> int:
        """Valida las entradas, predice y guarda el resultado."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado: {self.model_path}"
            )

        if not self.input_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el conjunto de prueba: {self.input_path}"
            )

        data = pd.read_csv(self.input_path)

        if data.empty:
            raise ValueError(
                f"El conjunto de prueba está vacío: {self.input_path}"
            )

        if data.columns.duplicated().any():
            duplicates = data.columns[
                data.columns.duplicated()
            ].tolist()
            raise ValueError(
                f"Hay columnas duplicadas en el conjunto de prueba: "
                f"{duplicates}"
            )

        # Se conserva el conjunto original y se excluye cualquier
        # columna objetivo existente de los predictores.
        X = data.drop(
            columns=[self.target, self.prediction_column],
            errors="ignore",
        ).copy()

        if X.shape[1] == 0:
            raise ValueError("No hay variables predictoras para predecir.")

        non_numeric = X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise ValueError(
                "Todos los predictores deben ser numéricos. "
                f"Columnas no numéricas: {non_numeric}"
            )

        X = X.astype(float)

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "El conjunto de prueba contiene valores faltantes "
                "o infinitos. No se imputan automáticamente."
            )

        # Carga el pipeline completo: selector de correlación y modelo.
        model = joblib.load(self.model_path)

        # Comprueba las variables esperadas por el pipeline y ajusta
        # el orden de las columnas si es necesario.
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            expected_features = list(expected_features)

            missing = sorted(set(expected_features) - set(X.columns))
            extra = sorted(set(X.columns) - set(expected_features))

            if missing or extra:
                raise ValueError(
                    "Las variables del conjunto de prueba no coinciden "
                    "con las esperadas por el modelo. "
                    f"Faltantes: {missing}. Adicionales: {extra}."
                )

            X = X.loc[:, expected_features]

        predictions = np.asarray(model.predict(X)).reshape(-1)

        if len(predictions) != len(data):
            raise ValueError(
                "El número de predicciones no coincide con el número "
                "de filas del conjunto de prueba."
            )

        if not np.isfinite(predictions).all():
            raise ValueError("El modelo produjo predicciones no finitas.")

        # Se agrega target_pred sin modificar el archivo de entrada.
        result = data.copy()
        result[self.prediction_column] = predictions

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result.to_csv(self.output_path, index=False)

        print(f"Filas procesadas: {len(result)}")
        print(f"Modelo utilizado: {self.model_path}")
        print(f"Archivo generado: {self.output_path}")
        print(f"Columna de predicción: {self.prediction_column}")

        return 0


def main() -> int:
    """Punto de entrada del script."""
    pipeline = PredictionPipeline(
        model_path=MODEL_PATH,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )
    return pipeline.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
