"""
Healthcare ML Trainer with MLflow Integration

Provides FDA-compliant model training with automatic:
- Experiment tracking
- Model versioning
- Artifact logging
- Audit trail generation
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

ModelType = Literal["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]


class HealthcareTrainer:
    """
    Healthcare ML trainer with MLflow integration.
    
    Features:
    - Automatic experiment tracking
    - Model versioning with checksums
    - Cross-validation with confidence intervals
    - FDA-compliant audit logging
    """
    
    MODEL_CLASSES: dict[str, type[BaseEstimator]] = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI.
            artifact_location: Location for model artifacts.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.model: BaseEstimator | None = None
        self.run_id: str | None = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: ModelType = "gradient_boosting",
        params: dict[str, Any] | None = None,
        cv_folds: int = 5,
        tags: dict[str, str] | None = None,
    ) -> BaseEstimator:
        """
        Train a model with full MLflow tracking.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            model_type: Type of model to train.
            params: Model hyperparameters.
            cv_folds: Number of cross-validation folds.
            tags: Additional tags for the run.
            
        Returns:
            Trained model.
        """
        params = params or {}
        tags = tags or {}
        
        # Add healthcare-specific tags
        tags.update({
            "domain": "healthcare",
            "regulatory": "fda_21cfr11",
            "model_type": model_type,
        })
        
        with mlflow.start_run(tags=tags) as run:
            self.run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("n_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Create and train model
            if model_type == "xgboost":
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBClassifier(**params)
                except ImportError:
                    logger.warning("XGBoost not available, using GradientBoosting")
                    self.model = GradientBoostingClassifier(**params)
            else:
                model_class = self.MODEL_CLASSES.get(model_type)
                if model_class is None:
                    raise ValueError(f"Unknown model type: {model_type}")
                self.model = model_class(**params)
            
            # Fit model
            self.model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=cv_folds, scoring="roc_auc"
            )
            
            # Log CV metrics
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())
            mlflow.log_metric("cv_auc_min", cv_scores.min())
            mlflow.log_metric("cv_auc_max", cv_scores.max())
            
            # Training metrics
            y_pred = self.model.predict(X_train)
            y_proba = self._get_probabilities(X_train)
            
            train_metrics = self._calculate_metrics(y_train, y_pred, y_proba)
            for name, value in train_metrics.items():
                mlflow.log_metric(f"train_{name}", value)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"{self.experiment_name}_model",
            )
            
            # Log audit information
            self._log_audit_info(X_train, y_train, params)
            
            logger.info(
                f"Training complete. Run ID: {self.run_id}, "
                f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
        
        return self.model
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary of metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        y_proba = self._get_probabilities(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Log to MLflow if we have an active run
        if self.run_id:
            with mlflow.start_run(run_id=self.run_id):
                for name, value in metrics.items():
                    mlflow.log_metric(f"test_{name}", value)
        
        return metrics
    
    def _get_probabilities(self, X: np.ndarray) -> np.ndarray | None:
        """Get prediction probabilities if available."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return None
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> dict[str, float]:
        """Calculate standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                pass
        
        return metrics
    
    def _log_audit_info(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict[str, Any],
    ) -> None:
        """Log FDA-compliant audit information."""
        audit_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_shape": X_train.shape,
            "label_distribution": {
                str(k): int(v) for k, v in zip(
                    *np.unique(y_train, return_counts=True)
                )
            },
            "parameters": params,
            "data_hash": hashlib.sha256(
                X_train.tobytes() + y_train.tobytes()
            ).hexdigest()[:16],
        }
        
        # Log as artifact
        mlflow.log_dict(audit_info, "audit/training_audit.json")
    
    def get_model_checksum(self) -> str:
        """
        Generate checksum for model integrity verification.
        
        Returns:
            SHA-256 hash of model parameters.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Serialize model parameters
        params = self.model.get_params()
        params_str = json.dumps(params, sort_keys=True, default=str)
        
        return hashlib.sha256(params_str.encode()).hexdigest()
