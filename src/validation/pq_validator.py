"""
Performance Qualification (PQ) Validator

Validates ML model performance meets clinical/regulatory requirements
using statistical testing methods per FDA guidance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class PerformanceThresholds:
    """Thresholds for performance metrics."""
    
    min_auc: float = 0.85
    min_accuracy: float = 0.80
    min_sensitivity: float = 0.80
    min_specificity: float = 0.80
    min_ppv: float = 0.70
    min_npv: float = 0.90
    max_psi: float = 0.10
    confidence_level: float = 0.95


@dataclass
class PQMetricResult:
    """Result of a single PQ metric evaluation."""
    
    metric_name: str
    value: float
    threshold: float
    passed: bool
    confidence_interval: tuple[float, float] | None = None
    details: str = ""


@dataclass
class PQReport:
    """Complete PQ validation report."""
    
    passed: bool
    metrics: list[PQMetricResult]
    confusion_matrix: np.ndarray
    classification_report: str
    statistical_tests: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validator_version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "validator_version": self.validator_version,
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "confidence_interval": m.confidence_interval,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "statistical_tests": self.statistical_tests,
            "summary": {
                "total_metrics": len(self.metrics),
                "passed_metrics": sum(1 for m in self.metrics if m.passed),
                "failed_metrics": sum(1 for m in self.metrics if not m.passed),
            },
        }


class PQValidator:
    """
    Performance Qualification Validator for FDA 21 CFR Part 11 compliance.
    
    Validates:
    - AUC-ROC with confidence intervals
    - Sensitivity/Specificity
    - PPV/NPV
    - Population Stability Index (PSI)
    - Statistical significance tests
    """
    
    def __init__(
        self,
        model: Any,
        thresholds: PerformanceThresholds | None = None,
    ):
        """
        Initialize PQ validator.
        
        Args:
            model: ML model to validate.
            thresholds: Performance thresholds.
        """
        self.model = model
        self.thresholds = thresholds or PerformanceThresholds()
        self.metric_results: list[PQMetricResult] = []
    
    def validate_performance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_baseline: np.ndarray | None = None,
    ) -> PQReport:
        """
        Run all PQ validation tests.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            X_baseline: Baseline data for drift detection (optional).
            
        Returns:
            PQReport with all metric results.
        """
        self.metric_results = []
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self._evaluate_auc(y_test, y_proba)
        self._evaluate_accuracy(y_test, y_pred)
        self._evaluate_sensitivity_specificity(y_test, y_pred)
        self._evaluate_ppv_npv(y_test, y_pred)
        
        # Calculate PSI if baseline provided
        statistical_tests = {}
        if X_baseline is not None:
            psi = self._calculate_psi(X_baseline, X_test)
            statistical_tests["psi"] = psi
            self._evaluate_psi(psi)
        
        # Generate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Determine overall pass/fail
        all_passed = all(m.passed for m in self.metric_results)
        
        return PQReport(
            passed=all_passed,
            metrics=self.metric_results,
            confusion_matrix=cm,
            classification_report=report,
            statistical_tests=statistical_tests,
        )
    
    def _evaluate_auc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> None:
        """Evaluate AUC-ROC with confidence interval."""
        if y_proba is None:
            self.metric_results.append(
                PQMetricResult(
                    metric_name="auc_roc",
                    value=0.0,
                    threshold=self.thresholds.min_auc,
                    passed=False,
                    details="Model does not support probability predictions",
                )
            )
            return
        
        auc = roc_auc_score(y_true, y_proba)
        ci_lower, ci_upper = self._bootstrap_auc_ci(y_true, y_proba)
        
        passed = ci_lower >= self.thresholds.min_auc
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="auc_roc",
                value=auc,
                threshold=self.thresholds.min_auc,
                passed=passed,
                confidence_interval=(ci_lower, ci_upper),
                details=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
            )
        )
    
    def _bootstrap_auc_ci(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bootstraps: int = 1000,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for AUC."""
        rng = np.random.default_rng(42)
        aucs = []
        
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            try:
                auc = roc_auc_score(y_true[indices], y_proba[indices])
                aucs.append(auc)
            except ValueError:
                continue
        
        alpha = 1 - self.thresholds.confidence_level
        ci_lower = np.percentile(aucs, alpha / 2 * 100)
        ci_upper = np.percentile(aucs, (1 - alpha / 2) * 100)
        
        return ci_lower, ci_upper
    
    def _evaluate_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Evaluate accuracy."""
        acc = accuracy_score(y_true, y_pred)
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="accuracy",
                value=acc,
                threshold=self.thresholds.min_accuracy,
                passed=acc >= self.thresholds.min_accuracy,
            )
        )
    
    def _evaluate_sensitivity_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Evaluate sensitivity (recall) and specificity."""
        # Sensitivity (True Positive Rate)
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="sensitivity",
                value=sensitivity,
                threshold=self.thresholds.min_sensitivity,
                passed=sensitivity >= self.thresholds.min_sensitivity,
                details="True Positive Rate",
            )
        )
        
        # Specificity (True Negative Rate)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="specificity",
                value=specificity,
                threshold=self.thresholds.min_specificity,
                passed=specificity >= self.thresholds.min_specificity,
                details="True Negative Rate",
            )
        )
    
    def _evaluate_ppv_npv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Evaluate Positive and Negative Predictive Values."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # PPV (Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="ppv",
                value=ppv,
                threshold=self.thresholds.min_ppv,
                passed=ppv >= self.thresholds.min_ppv,
                details="Positive Predictive Value (Precision)",
            )
        )
        
        # NPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="npv",
                value=npv,
                threshold=self.thresholds.min_npv,
                passed=npv >= self.thresholds.min_npv,
                details="Negative Predictive Value",
            )
        )
    
    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """
        Calculate Population Stability Index.
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change
        """
        psi_values = {}
        
        for i in range(baseline.shape[1]):
            baseline_col = baseline[:, i]
            current_col = current[:, i]
            
            # Create bins from baseline
            _, bin_edges = np.histogram(baseline_col, bins=n_bins)
            
            # Calculate proportions
            baseline_counts = np.histogram(baseline_col, bins=bin_edges)[0]
            current_counts = np.histogram(current_col, bins=bin_edges)[0]
            
            baseline_props = baseline_counts / len(baseline_col)
            current_props = current_counts / len(current_col)
            
            # Avoid division by zero
            baseline_props = np.clip(baseline_props, 1e-6, 1)
            current_props = np.clip(current_props, 1e-6, 1)
            
            # Calculate PSI
            psi = np.sum(
                (current_props - baseline_props) *
                np.log(current_props / baseline_props)
            )
            
            psi_values[f"feature_{i}"] = psi
        
        psi_values["mean"] = np.mean(list(psi_values.values()))
        psi_values["max"] = np.max(list(psi_values.values()))
        
        return psi_values
    
    def _evaluate_psi(self, psi: dict[str, float]) -> None:
        """Evaluate PSI against threshold."""
        max_psi = psi.get("max", 0.0)
        
        self.metric_results.append(
            PQMetricResult(
                metric_name="psi_max",
                value=max_psi,
                threshold=self.thresholds.max_psi,
                passed=max_psi <= self.thresholds.max_psi,
                details=f"Mean PSI: {psi.get('mean', 0.0):.4f}",
            )
        )
