"""
Healthcare Model Registry with Approval Workflows

Production model registry supporting:
- Model versioning with lineage
- Stage transitions (Dev → Staging → Production)
- Approval workflows for regulated environments
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NLP = "nlp"
    LLM = "llm"


@dataclass
class ModelMetrics:
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "f1_score": self.f1_score, "auc_roc": self.auc_roc}.items() if v is not None}


@dataclass
class ModelLineage:
    training_data_uri: str
    training_data_version: str
    training_data_hash: str
    training_run_id: str
    training_start_time: datetime
    training_end_time: datetime
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredModel:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    artifact_uri: str
    metrics: ModelMetrics
    lineage: ModelLineage
    created_at: datetime
    created_by: str


class ModelRegistryConfig(BaseModel):
    require_approval_for_production: bool = True
    approval_expiry_hours: int = Field(default=72, ge=1)
    min_metrics_for_promotion: dict[str, float] = Field(default_factory=lambda: {"accuracy": 0.8})


class ModelRegistry:
    """Healthcare Model Registry with approval workflows."""

    def __init__(self, config: ModelRegistryConfig | None = None):
        self.config = config or ModelRegistryConfig()
        self._models: dict[str, RegisteredModel] = {}

    def register_model(self, name: str, version: str, model_type: ModelType, artifact_uri: str, metrics: ModelMetrics, lineage: ModelLineage, created_by: str = "system") -> RegisteredModel:
        model_id = f"model_{hashlib.sha256(f'{name}:{version}'.encode()).hexdigest()[:16]}"
        model = RegisteredModel(model_id=model_id, name=name, version=version, model_type=model_type, stage=ModelStage.DEVELOPMENT, artifact_uri=artifact_uri, metrics=metrics, lineage=lineage, created_at=datetime.utcnow(), created_by=created_by)
        self._models[model_id] = model
        return model

    def get_model(self, model_id: str) -> RegisteredModel | None:
        return self._models.get(model_id)

    def list_models(self, stage: ModelStage | None = None) -> list[RegisteredModel]:
        results = list(self._models.values())
        if stage:
            results = [m for m in results if m.stage == stage]
        return results
