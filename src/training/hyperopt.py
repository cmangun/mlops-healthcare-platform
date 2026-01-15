"""
Hyperparameter Optimization with MLflow Tracking

Provides Optuna-based hyperparameter optimization with full
experiment tracking for healthcare ML models.
"""

import logging
from typing import Any, Callable

import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer with MLflow integration.
    
    Uses Optuna for efficient hyperparameter search with
    automatic logging of all trials to MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        param_space: dict[str, Any],
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        direction: str = "maximize",
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features.
            y: Training labels.
            model_class: Sklearn-compatible model class.
            param_space: Parameter search space definition.
            n_trials: Number of optimization trials.
            cv_folds: Cross-validation folds.
            scoring: Scoring metric.
            direction: Optimization direction.
            
        Returns:
            Best parameters found.
        """
        try:
            import optuna
            from optuna.integration.mlflow import MLflowCallback
        except ImportError:
            logger.warning("Optuna not installed, using random search")
            return self._random_search(
                X, y, model_class, param_space, n_trials, cv_folds, scoring
            )
        
        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"]
                    )
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(
                        name, spec["choices"]
                    )
            
            # Train and evaluate
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction=direction)
        
        # Add MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.tracking_uri,
            metric_name=scoring,
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )
        
        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def _random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: type,
        param_space: dict[str, Any],
        n_trials: int,
        cv_folds: int,
        scoring: str,
    ) -> dict[str, Any]:
        """Fallback random search when Optuna not available."""
        best_score = -np.inf
        best_params = {}
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "int":
                    params[name] = np.random.randint(
                        spec["low"], spec["high"] + 1
                    )
                elif spec["type"] == "float":
                    if spec.get("log", False):
                        params[name] = np.exp(
                            np.random.uniform(
                                np.log(spec["low"]),
                                np.log(spec["high"])
                            )
                        )
                    else:
                        params[name] = np.random.uniform(
                            spec["low"], spec["high"]
                        )
                elif spec["type"] == "categorical":
                    params[name] = np.random.choice(spec["choices"])
            
            # Evaluate
            with mlflow.start_run(nested=True):
                model = model_class(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=scoring
                )
                score = scores.mean()
                
                mlflow.log_params(params)
                mlflow.log_metric(scoring, score)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return best_params


# Common parameter spaces for healthcare models
PARAM_SPACES = {
    "gradient_boosting": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
}
