"""Construcción de la búsqueda de modelos de regresión."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ModelSearch:
    """Construye la búsqueda de hiperparámetros sin ejecutar el entrenamiento."""

    @staticmethod
    def build_search(
        selector_class,
        min_abs_correlation: float,
        max_abs_correlation: float,
        random_state: int,
        cv_splits: int,
        n_jobs: int,
    ) -> GridSearchCV:
        """Crea el pipeline y GridSearchCV con la configuración del proyecto."""
        pipeline = Pipeline([
            (
                "selector",
                selector_class(
                    min_abs_correlation,
                    max_abs_correlation,
                ),
            ),
            ("scaler", "passthrough"),
            ("model", LinearRegression()),
        ])

        grids = [
            # Referencia: regresión lineal.
            {
                "model": [LinearRegression()],
                "scaler": [StandardScaler()],
            },
            # Ridge: siete valores de regularización.
            {
                "model": [Ridge()],
                "scaler": [StandardScaler()],
                "model__alpha": [
                    0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
                ],
            },
            # Random Forest: 12 configuraciones.
            {
                "model": [
                    RandomForestRegressor(
                        random_state=random_state,
                        n_jobs=1,
                    )
                ],
                "scaler": ["passthrough"],
                "model__n_estimators": [200, 300],
                "model__max_depth": [5, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
            # XGBoost: 108 configuraciones.
            {
                "model": [
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                        verbosity=0,
                    )
                ],
                "scaler": ["passthrough"],
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [1, 2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.10],
                "model__min_child_weight": [1, 3, 5],
            },
        ]

        return GridSearchCV(
            pipeline,
            param_grid=grids,
            scoring="neg_root_mean_squared_error",
            cv=KFold(
                n_splits=cv_splits,
                shuffle=True,
                random_state=random_state,
            ),
            refit=False,
            return_train_score=True,
            n_jobs=n_jobs,
            pre_dispatch=n_jobs,
            error_score="raise",
            verbose=1,
        )
