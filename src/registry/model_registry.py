"""
Healthcare Model Registry with Approval Workflows

Production model registry supporting:
- Model versioning with lineage
- Stage transitions (Dev → Staging → Production)
- Approval workflows for regulated environments
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    training_start_time: datetime = field(default_factory=datetime.utcnow)
    training_end_time: datetime = field(default_factory=datetime.utcnow)
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredModel:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    artifact_uri: str
    metrics: ModelMetrics | None = None
    lineage: ModelLineage | None = None
    model_card: "ModelCard" | None = None
    created_at: datetime
    created_by: str
    last_updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelCard:
    model_name: str
    model_version: str
    model_type: str
    description: str
    intended_use: str
    out_of_scope_uses: list[str]
    training_data_description: str
    evaluation_data_description: str
    ethical_considerations: str
    clinical_validation_status: str
    regulatory_status: str


@dataclass
class PromotionRequest:
    request_id: str
    model_id: str
    from_stage: ModelStage
    to_stage: ModelStage
    requester_id: str
    justification: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=72))
    approver_id: str | None = None
    approval_comments: str | None = None


class ModelRegistryConfig(BaseModel):
    require_approval_for_production: bool = True
    approval_expiry_hours: int = Field(default=72, ge=1)
    min_metrics_for_promotion: dict[str, float] = Field(default_factory=lambda: {"accuracy": 0.8})
    min_accuracy_for_production: float = Field(default=0.85, ge=0.0, le=1.0)


class ModelRegistry:
    """Healthcare Model Registry with approval workflows."""

    def __init__(self, config: ModelRegistryConfig | None = None):
        self.config = config or ModelRegistryConfig()
        self._models: dict[str, RegisteredModel] = {}
        self._promotion_requests: dict[str, PromotionRequest] = {}

    def register_model(
        self,
        name: str,
        version: str,
        model_type: ModelType,
        artifact_uri: str,
        metrics: ModelMetrics | None = None,
        lineage: ModelLineage | None = None,
        model_card: ModelCard | None = None,
        created_by: str = "system",
    ) -> RegisteredModel:
        model_id = f"model_{hashlib.sha256(f'{name}:{version}'.encode()).hexdigest()[:16]}"
        model = RegisteredModel(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            stage=ModelStage.DEVELOPMENT,
            artifact_uri=artifact_uri,
            metrics=metrics,
            lineage=lineage,
            model_card=model_card,
            created_at=datetime.utcnow(),
            created_by=created_by,
        )
        self._models[model_id] = model
        return model

    def get_model(self, model_id: str) -> RegisteredModel:
        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        return model

    def list_models(self, stage: ModelStage | None = None) -> list[RegisteredModel]:
        results = list(self._models.values())
        if stage:
            results = [m for m in results if m.stage == stage]
        return results

    def request_promotion(
        self,
        model_id: str,
        to_stage: ModelStage,
        requester_id: str,
        justification: str,
    ) -> PromotionRequest:
        model = self.get_model(model_id)
        self._validate_promotion_requirements(model, to_stage)

        request_id = f"req_{hashlib.sha256(f'{model_id}:{to_stage}:{requester_id}:{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:16]}"
        request = PromotionRequest(
            request_id=request_id,
            model_id=model_id,
            from_stage=model.stage,
            to_stage=to_stage,
            requester_id=requester_id,
            justification=justification,
            expires_at=datetime.utcnow() + timedelta(hours=self.config.approval_expiry_hours),
        )
        self._promotion_requests[request_id] = request
        return request

    def approve_promotion(self, request_id: str, approver_id: str, comments: str | None = None) -> PromotionRequest:
        request = self._promotion_requests.get(request_id)
        if not request:
            raise ValueError(f"Promotion request {request_id} not found")
        if request.status != ApprovalStatus.PENDING:
            raise ValueError("Promotion request is not pending")
        if datetime.utcnow() > request.expires_at:
            raise ValueError("Promotion request expired")

        model = self.get_model(request.model_id)
        model.stage = request.to_stage
        model.last_updated_at = datetime.utcnow()

        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approval_comments = comments
        return request

    def get_latest_model(self, name: str) -> RegisteredModel:
        candidates = [model for model in self._models.values() if model.name == name]
        if not candidates:
            raise ValueError(f"No models found for name {name}")
        return max(candidates, key=lambda model: self._version_key(model.version))

    def _validate_promotion_requirements(self, model: RegisteredModel, to_stage: ModelStage) -> None:
        if to_stage != ModelStage.PRODUCTION:
            return
        if not self.config.require_approval_for_production:
            return
        if not model.metrics or model.metrics.accuracy is None:
            raise ValueError("accuracy metric required for production promotion")
        if model.metrics.accuracy < self.config.min_accuracy_for_production:
            raise ValueError("accuracy below minimum for production promotion")

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        parts = version.split(".")
        parsed: list[int] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError:
                parsed.append(0)
        return tuple(parsed)
