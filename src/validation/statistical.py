"""
Statistical Validation Suite

Comprehensive statistical tests for ML model validation
in regulated healthcare environments.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    interpretation: str
    details: dict[str, Any] | None = None


class StatisticalValidator:
    """
    Statistical validation suite for healthcare ML models.
    
    Provides tests for:
    - Distribution comparison (KS test, Chi-square)
    - Performance stability (PSI, CSI)
    - Calibration assessment
    - Fairness metrics
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize validator.
        
        Args:
            significance_level: Alpha for hypothesis tests.
        """
        self.alpha = significance_level
    
    def ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> StatisticalTestResult:
        """
        Kolmogorov-Smirnov test for distribution comparison.
        
        Tests if two samples come from the same distribution.
        
        Args:
            baseline: Baseline distribution sample.
            current: Current distribution sample.
            
        Returns:
            StatisticalTestResult with KS statistic and p-value.
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        passed = p_value > self.alpha
        
        if statistic < 0.1:
            interpretation = "No significant distribution shift"
        elif statistic < 0.2:
            interpretation = "Moderate distribution shift detected"
        else:
            interpretation = "Significant distribution shift detected"
        
        return StatisticalTestResult(
            test_name="kolmogorov_smirnov",
            statistic=statistic,
            p_value=p_value,
            passed=passed,
            interpretation=interpretation,
            details={
                "baseline_mean": np.mean(baseline),
                "current_mean": np.mean(current),
                "baseline_std": np.std(baseline),
                "current_std": np.std(current),
            },
        )
    
    def chi_square_test(
        self,
        observed: np.ndarray,
        expected: np.ndarray,
    ) -> StatisticalTestResult:
        """
        Chi-square test for categorical distribution comparison.
        
        Args:
            observed: Observed frequencies.
            expected: Expected frequencies.
            
        Returns:
            StatisticalTestResult with chi-square statistic and p-value.
        """
        # Normalize to same total
        observed = observed * (expected.sum() / observed.sum())
        
        statistic, p_value = stats.chisquare(observed, expected)
        
        passed = p_value > self.alpha
        
        return StatisticalTestResult(
            test_name="chi_square",
            statistic=statistic,
            p_value=p_value,
            passed=passed,
            interpretation="Distributions are similar" if passed
            else "Distributions significantly different",
        )
    
    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change, investigation needed
        - PSI >= 0.25: Significant change, action required
        
        Args:
            baseline: Baseline distribution.
            current: Current distribution.
            n_bins: Number of bins.
            
        Returns:
            PSI value.
        """
        # Create bins from baseline
        _, bin_edges = np.histogram(baseline, bins=n_bins)
        
        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]
        
        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)
        
        # Avoid division by zero
        baseline_props = np.clip(baseline_props, 1e-6, 1)
        current_props = np.clip(current_props, 1e-6, 1)
        
        # Calculate PSI
        psi = np.sum(
            (current_props - baseline_props) *
            np.log(current_props / baseline_props)
        )
        
        return psi
    
    def auc_with_ci(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
    ) -> tuple[float, float, float]:
        """
        Calculate AUC-ROC with bootstrap confidence interval.
        
        Args:
            y_true: True binary labels.
            y_score: Predicted scores/probabilities.
            confidence: Confidence level.
            n_bootstraps: Number of bootstrap samples.
            
        Returns:
            Tuple of (AUC, CI lower, CI upper).
        """
        from sklearn.metrics import roc_auc_score
        
        auc = roc_auc_score(y_true, y_score)
        
        rng = np.random.default_rng(42)
        aucs = []
        
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            try:
                boot_auc = roc_auc_score(y_true[indices], y_score[indices])
                aucs.append(boot_auc)
            except ValueError:
                continue
        
        alpha = 1 - confidence
        ci_lower = np.percentile(aucs, alpha / 2 * 100)
        ci_upper = np.percentile(aucs, (1 - alpha / 2) * 100)
        
        return auc, ci_lower, ci_upper
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> float:
        """
        Calculate demographic parity ratio.
        
        Measures if positive prediction rates are equal across groups.
        
        Args:
            y_pred: Predicted labels.
            sensitive_attr: Sensitive attribute (binary).
            
        Returns:
            Demographic parity ratio (0-1, closer to 1 is fairer).
        """
        groups = np.unique(sensitive_attr)
        if len(groups) != 2:
            raise ValueError("Sensitive attribute must be binary")
        
        rates = []
        for group in groups:
            mask = sensitive_attr == group
            rate = np.mean(y_pred[mask])
            rates.append(rate)
        
        # Ratio of min to max rate
        return min(rates) / max(rates) if max(rates) > 0 else 0.0
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate equalized odds metrics.
        
        Measures if TPR and FPR are equal across groups.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_attr: Sensitive attribute (binary).
            
        Returns:
            Dict with TPR and FPR ratios per group.
        """
        groups = np.unique(sensitive_attr)
        
        tprs = {}
        fprs = {}
        
        for group in groups:
            mask = sensitive_attr == group
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
            
            # TPR
            pos_mask = y_true_g == 1
            if pos_mask.sum() > 0:
                tprs[str(group)] = np.mean(y_pred_g[pos_mask])
            else:
                tprs[str(group)] = 0.0
            
            # FPR
            neg_mask = y_true_g == 0
            if neg_mask.sum() > 0:
                fprs[str(group)] = np.mean(y_pred_g[neg_mask])
            else:
                fprs[str(group)] = 0.0
        
        tpr_values = list(tprs.values())
        fpr_values = list(fprs.values())
        
        return {
            "tpr_ratio": min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0.0,
            "fpr_ratio": min(fpr_values) / max(fpr_values) if max(fpr_values) > 0 else 0.0,
            "tprs": tprs,
            "fprs": fprs,
        }
    
    def calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Tuple of (ECE, mean predicted per bin, fraction positive per bin).
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        mean_predicted = []
        fraction_positive = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                mean_predicted.append(np.mean(y_proba[mask]))
                fraction_positive.append(np.mean(y_true[mask]))
                bin_counts.append(mask.sum())
            else:
                mean_predicted.append(0)
                fraction_positive.append(0)
                bin_counts.append(0)
        
        mean_predicted = np.array(mean_predicted)
        fraction_positive = np.array(fraction_positive)
        bin_counts = np.array(bin_counts)
        
        # ECE = weighted average of |predicted - actual|
        total = bin_counts.sum()
        ece = np.sum(
            bin_counts / total * np.abs(mean_predicted - fraction_positive)
        )
        
        return ece, mean_predicted, fraction_positive
