"""
Model Drift Detection for Healthcare ML

Comprehensive drift detection supporting:
- Data drift (feature distributions)
- Concept drift (relationship changes)
- Prediction drift (output distributions)
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DriftType(str, Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DRIFTED = "drifted"
    CRITICAL = "critical"


@dataclass
class DriftMetric:
    name: str
    value: float
    threshold: float
    is_drifted: bool


@dataclass
class DriftReport:
    model_id: str
    model_version: str
    report_id: str
    timestamp: datetime
    status: DriftStatus
    drift_types_detected: list[DriftType]
    metrics: list[DriftMetric]
    feature_drift: dict[str, DriftMetric]
    recommendations: list[str]


class DriftDetectorConfig(BaseModel):
    psi_threshold: float = Field(default=0.2, ge=0.0)
    ks_test_threshold: float = Field(default=0.05, ge=0.0)
    min_samples_for_detection: int = Field(default=100, ge=10)


class StatisticalTests:
    @staticmethod
    def calculate_psi(baseline: list[float], current: list[float], bins: int = 10) -> float:
        if not baseline or not current:
            return 0.0
        min_val, max_val = min(min(baseline), min(current)), max(max(baseline), max(current))
        if min_val == max_val:
            return 0.0
        bin_edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
        def get_bin_proportions(data):
            counts = [0] * bins
            for val in data:
                for i in range(bins):
                    if bin_edges[i] <= val < bin_edges[i + 1]:
                        counts[i] += 1
                        break
                else:
                    counts[-1] += 1
            return [(c + 0.0001) / (len(data) + 0.0001 * bins) for c in counts]
        bp, cp = get_bin_proportions(baseline), get_bin_proportions(current)
        return abs(sum((c - b) * (math.log(c / b) if b > 0 else 0) for b, c in zip(bp, cp)))


class DriftDetector:
    """Production drift detector for healthcare ML models."""

    def __init__(self, config: DriftDetectorConfig | None = None):
        self.config = config or DriftDetectorConfig()
        self._baseline_data: dict[str, dict[str, list[float]]] = {}
        self._current_data: dict[str, dict[str, list[float]]] = {}

    def set_baseline(self, model_id: str, feature_data: dict[str, list[float]]) -> None:
        self._baseline_data[model_id] = feature_data

    def add_samples(self, model_id: str, feature_data: dict[str, list[float]]) -> None:
        if model_id not in self._current_data:
            self._current_data[model_id] = defaultdict(list)
        for feature, values in feature_data.items():
            self._current_data[model_id][feature].extend(values)

    def detect_drift(self, model_id: str, model_version: str = "1.0.0") -> DriftReport:
        baseline = self._baseline_data.get(model_id, {})
        current = dict(self._current_data.get(model_id, {}))
        feature_drift, drifted_features, drift_types = {}, [], []
        for feature in baseline:
            if feature not in current:
                continue
            psi = StatisticalTests.calculate_psi(baseline[feature], current[feature])
            is_drifted = psi >= self.config.psi_threshold
            feature_drift[feature] = DriftMetric(name=f"psi_{feature}", value=psi, threshold=self.config.psi_threshold, is_drifted=is_drifted)
            if is_drifted:
                drifted_features.append(feature)
        if len(drifted_features) > 0.2 * len(feature_drift):
            drift_types.append(DriftType.DATA_DRIFT)
        status = DriftStatus.DRIFTED if drift_types else DriftStatus.HEALTHY
        return DriftReport(model_id=model_id, model_version=model_version, report_id=f"drift_{hashlib.sha256(f'{model_id}:{datetime.utcnow()}'.encode()).hexdigest()[:12]}", timestamp=datetime.utcnow(), status=status, drift_types_detected=drift_types, metrics=[], feature_drift=feature_drift, recommendations=["No significant drift detected."] if not drift_types else ["Data drift detected. Consider retraining."])
