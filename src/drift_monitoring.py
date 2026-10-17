
"""Monitorización de deriva de datos mediante el índice PSI."""

from pathlib import Path

import numpy as np
import pandas as pd


class DataDriftMonitor:
    """Compara las distribuciones de datos de referencia y datos recientes."""

    TARGET_COLUMN = "target"
    DEFAULT_N_BINS = 10
    DEFAULT_PSI_WARNING = 0.20
    DEFAULT_PSI_ALERT = 0.30
    EPSILON = 1e-6

    def __init__(
        self,
        reference_path: Path,
        current_path: Path,
        report_path: Path,
        n_bins: int = DEFAULT_N_BINS,
        psi_warning: float = DEFAULT_PSI_WARNING,
        psi_alert: float = DEFAULT_PSI_ALERT,
    ) -> None:
        """Inicializa las rutas y los parámetros de monitorización."""
        if n_bins < 2:
            raise ValueError("El número de intervalos debe ser al menos 2.")

        if not 0 <= psi_warning < psi_alert:
            raise ValueError(
                "Los umbrales deben cumplir: "
                "0 <= psi_warning < psi_alert."
            )

        self.reference_path = reference_path
        self.current_path = current_path
        self.report_path = report_path
        self.n_bins = n_bins
        self.psi_warning = psi_warning
        self.psi_alert = psi_alert

    def _load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carga y valida la estructura de los conjuntos de datos."""
        reference_df = pd.read_csv(self.reference_path)
        current_df = pd.read_csv(self.current_path)

        # Excluimos la variable objetivo de ambos conjuntos.
        reference_df = reference_df.drop(
            columns=[self.TARGET_COLUMN],
            errors="ignore",
        )

        current_df = current_df.drop(
            columns=[self.TARGET_COLUMN],
            errors="ignore",
        )

        if reference_df.empty or current_df.empty:
            raise ValueError(
                "Ambos conjuntos deben contener observaciones."
            )

        if reference_df.columns.has_duplicates:
            raise ValueError(
                "El conjunto de referencia contiene columnas duplicadas."
            )

        if current_df.columns.has_duplicates:
            raise ValueError(
                "El conjunto reciente contiene columnas duplicadas."
            )

        # Comprobamos que ambos conjuntos tengan las mismas características.
        reference_columns = set(reference_df.columns)
        current_columns = set(current_df.columns)

        if reference_columns != current_columns:
            missing = sorted(reference_columns - current_columns)
            unexpected = sorted(current_columns - reference_columns)

            raise ValueError(
                f"El esquema de características no coincide. "
                f"Columnas faltantes: {missing}; "
                f"columnas inesperadas: {unexpected}"
            )

        # Conservamos el mismo orden de las características.
        current_df = current_df[reference_df.columns]

        return reference_df, current_df

    def _validate_feature(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature: str,
    ) -> tuple[pd.Series, pd.Series]:
        """Comprueba que una característica sea numérica y válida."""
        try:
            reference = pd.to_numeric(reference, errors="raise")
            current = pd.to_numeric(current, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"La característica '{feature}' no es numérica."
            ) from exc

        if reference.isna().any() or current.isna().any():
            raise ValueError(
                f"Se detectaron valores faltantes en '{feature}'."
            )

        # Comprobamos que no existan infinitos ni otros valores no finitos.
        reference_values = reference.to_numpy(dtype=float)
        current_values = current.to_numpy(dtype=float)

        if not np.isfinite(reference_values).all():
            raise ValueError(
                f"La referencia contiene valores no finitos en '{feature}'."
            )

        if not np.isfinite(current_values).all():
            raise ValueError(
                f"Los datos recientes contienen valores no finitos "
                f"en '{feature}'."
            )

        return reference, current

    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calcula el PSI usando intervalos definidos por la referencia."""
        # Convertimos las series en arrays sin reasignar las variables
        # declaradas como pd.Series.
        reference_values = reference.to_numpy(dtype=float)
        current_values = current.to_numpy(dtype=float)

        if len(reference_values) == 0 or len(current_values) == 0:
            raise ValueError(
                "Las series deben contener observaciones."
            )

        # Definimos los intervalos a partir de los cuantiles de referencia.
        quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        inner_edges = np.unique(
            np.quantile(reference_values, quantiles)
        )

        # Los extremos infinitos permiten incluir todos los valores.
        edges = np.concatenate(
            ([-np.inf], inner_edges, [np.inf])
        )

        reference_counts, _ = np.histogram(
            reference_values, bins=edges
        )
        current_counts, _ = np.histogram(
            current_values, bins=edges
        )

        # Calculamos las proporciones de observaciones por intervalo.
        reference_pct = reference_counts / len(reference_values)
        current_pct = current_counts / len(current_values)

        # Evitamos ceros y normalizamos las proporciones suavizadas.
        reference_pct = np.maximum(
            reference_pct, self.EPSILON
        )
        current_pct = np.maximum(
            current_pct, self.EPSILON
        )

        reference_pct = reference_pct / reference_pct.sum()
        current_pct = current_pct / current_pct.sum()

        # Aplicamos la fórmula del índice de estabilidad de la población.
        psi = np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )

        return float(psi)

    def classify_psi(self, psi: float) -> str:
        """Clasifica el resultado según los umbrales configurados."""
        if psi >= self.psi_alert:
            return "ALERTA"

        if psi >= self.psi_warning:
            return "INVESTIGAR"

        return "SIN_ALERTA"

    def generate_report(self) -> pd.DataFrame:
        """Calcula el PSI de cada característica y guarda el informe."""
        reference_df, current_df = self._load_datasets()

        results: list[dict[str, str | float]] = []

        for feature in reference_df.columns:
            reference, current = self._validate_feature(
                reference_df[feature],
                current_df[feature],
                feature,
            )

            psi = self.calculate_psi(reference, current)

            results.append(
                {
                    "feature": feature,
                    "psi": psi,
                    "status": self.classify_psi(psi),
                }
            )

        # Ordenamos las características desde el mayor PSI al menor.
        report = pd.DataFrame(results).sort_values(
            "psi",
            ascending=False,
        )

        # Creamos la carpeta de destino si todavía no existe.
        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report.to_csv(self.report_path, index=False)

        return report


def main() -> None:
    """Ejecuta el análisis de deriva de datos."""
    project_root = Path(__file__).resolve().parents[1]

    monitor = DataDriftMonitor(
        reference_path=project_root / "data" / "training_data.csv",
        current_path=project_root / "data" / "blind_test_data.csv",
        report_path=project_root / "reports" / "data_drift_psi.csv",
    )

    report = monitor.generate_report()

    print("\nInforme de deriva de datos")
    print(report.to_string(index=False))
    print(f"\nInforme guardado en: {monitor.report_path}")
    print(
        "\nNota: el PSI detecta cambios de distribución, "
        "pero no mide directamente el error ni demuestra "
        "una degradación del modelo."
    )


if __name__ == "__main__":
    main()
