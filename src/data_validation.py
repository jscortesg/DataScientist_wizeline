"""Validación del conjunto de datos para el entrenamiento de regresión."""

import numpy as np
import pandas as pd


class DatasetValidator:
    """Valida los datos de entrada y separa predictores y variable objetivo."""

    def __init__(
        self,
        target: str,
        data_path: str,
        cv_splits: int,
    ) -> None:
        self.target = target
        self.data_path = data_path
        self.cv_splits = cv_splits

    def validate(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Valida el DataFrame y devuelve X (predictores) e y (objetivo)."""
        if data.empty:
            raise ValueError(f"El archivo está vacío: {self.data_path}")

        if data.columns.duplicated().any():
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(f"Hay columnas duplicadas: {duplicates}")

        if self.target not in data.columns:
            raise ValueError(
                f"No existe la columna objetivo '{self.target}' "
                f"en {self.data_path}."
            )

        y = pd.to_numeric(data[self.target], errors="coerce").astype(float)
        raw_X = data.drop(columns=[self.target])

        if raw_X.shape[1] == 0:
            raise ValueError(
                f"No hay predictores aparte de '{self.target}'."
            )

        non_numeric = raw_X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise ValueError(
                "Todos los predictores deben ser numéricos. "
                f"Columnas no numéricas: {non_numeric}"
            )

        X = raw_X.astype(float)

        # Comprobaciones de valores faltantes e infinitos.
        missing_mask = X.isna().any().to_numpy()
        infinite_mask = ~np.isfinite(X.to_numpy()).all(axis=0)

        bad_columns = X.columns[
            missing_mask | infinite_mask
        ].tolist()

        if bad_columns:
            raise ValueError(
                "Predictores con valores faltantes o infinitos: "
                f"{bad_columns}. No se imputan automáticamente."
            )

        if y.isna().any() or not np.isfinite(y.to_numpy()).all():
            raise ValueError(
                f"'{self.target}' contiene valores faltantes, "
                "no numéricos o infinitos."
            )

        if len(data) < self.cv_splits + 2:
            raise ValueError(
                "No hay suficientes filas para el holdout "
                f"y CV de {self.cv_splits} folds."
            )

        return X, y
