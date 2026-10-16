import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class PearsonCorrelationSelector(BaseEstimator, TransformerMixin):
    """Selecciona predictores por correlación; fit ocurre dentro de cada fold."""

    def __init__(self, min_abs_correlation=0.10, max_abs_correlation=1.0, enabled=True):
        self.min_abs_correlation = min_abs_correlation
        self.max_abs_correlation = max_abs_correlation
        self.enabled = enabled

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        if not self.enabled:
            self.selected_features_ = list(X.columns)
            return self

        target = pd.Series(np.asarray(y).reshape(-1), index=X.index)

        correlations = X.corrwith(target).replace([np.inf, -np.inf], np.nan)
        absolute = correlations.abs()
        selected = absolute[
            absolute.ge(self.min_abs_correlation)
            & absolute.le(self.max_abs_correlation)
        ].index.tolist()
        if not selected:
            raise ValueError(
                "Ningún predictor cumple el umbral de correlación absoluta "
                f"[{self.min_abs_correlation}, {self.max_abs_correlation}] en "
                "una partición de entrenamiento. Revisa los umbrales o los datos; "
                "no se seleccionarán variables automáticamente como alternativa."
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.selected_features_ = selected
        self.correlations_ = correlations
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Faltan variables requeridas: {missing}")
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self):
        return np.asarray(self.selected_features_, dtype=object)