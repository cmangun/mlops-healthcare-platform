"""Tests for model registry module."""

import pytest
from datetime import datetime

from src.registry.model_registry import (
    ModelRegistry,
    ModelRegistryConfig,
    ModelType,
    ModelStage,
    ModelMetrics,
    ModelLineage,
    ModelCard,
    ApprovalStatus,
)


class TestModelRegistration:
    """Test model registration functionality."""
    
    def test_register_model_basic(self):
        """Test basic model registration."""
        registry = ModelRegistry()
        
        model = registry.register_model(
            name="diabetes-classifier",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/diabetes/v1.0.0",
            created_by="ml_engineer@test.com",
        )
        
        assert model.name == "diabetes-classifier"
        assert model.version == "1.0.0"
        assert model.model_type == ModelType.CLASSIFICATION
        assert model.stage == ModelStage.DEVELOPMENT
    
    def test_register_model_with_metrics(self):
        """Test model registration with metrics."""
        registry = ModelRegistry()
        
        metrics = ModelMetrics(
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.915,
            auc_roc=0.96,
        )
        
        model = registry.register_model(
            name="risk-predictor",
            version="2.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/risk/v2.0.0",
            metrics=metrics,
            created_by="data_scientist@test.com",
        )
        
        assert model.metrics.accuracy == 0.92
        assert model.metrics.auc_roc == 0.96
    
    def test_register_model_with_lineage(self):
        """Test model registration with lineage tracking."""
        registry = ModelRegistry()
        
        lineage = ModelLineage(
            training_data_uri="s3://data/training/v3",
            training_data_version="3.0.0",
            training_data_hash="sha256:abc123",
            training_run_id="run_456",
            hyperparameters={"learning_rate": 0.01, "epochs": 100},
        )
        
        model = registry.register_model(
            name="outcome-predictor",
            version="1.0.0",
            model_type=ModelType.REGRESSION,
            artifact_uri="s3://models/outcome/v1.0.0",
            lineage=lineage,
            created_by="ml_team@test.com",
        )
        
        assert model.lineage.training_data_version == "3.0.0"
        assert model.lineage.hyperparameters["learning_rate"] == 0.01


class TestModelPromotion:
    """Test model stage transitions."""
    
    def test_promote_to_staging(self):
        """Test promotion from development to staging."""
        registry = ModelRegistry()
        
        model = registry.register_model(
            name="test-model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/test/v1.0.0",
            created_by="engineer@test.com",
        )
        
        # Request promotion
        request = registry.request_promotion(
            model_id=model.model_id,
            to_stage=ModelStage.STAGING,
            requester_id="engineer@test.com",
            justification="Ready for staging validation",
        )
        
        assert request.from_stage == ModelStage.DEVELOPMENT
        assert request.to_stage == ModelStage.STAGING
        assert request.status == ApprovalStatus.PENDING
    
    def test_approve_promotion(self):
        """Test promotion approval workflow."""
        registry = ModelRegistry()
        
        model = registry.register_model(
            name="approved-model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/approved/v1.0.0",
            metrics=ModelMetrics(accuracy=0.90),
            created_by="engineer@test.com",
        )
        
        # Request and approve
        request = registry.request_promotion(
            model_id=model.model_id,
            to_stage=ModelStage.STAGING,
            requester_id="engineer@test.com",
            justification="Metrics exceed threshold",
        )
        
        registry.approve_promotion(
            request_id=request.request_id,
            approver_id="ml_lead@test.com",
            comments="Approved after review",
        )
        
        # Check model was promoted
        updated_model = registry.get_model(model.model_id)
        assert updated_model.stage == ModelStage.STAGING
    
    def test_production_requires_metrics(self):
        """Test that production promotion requires minimum metrics."""
        config = ModelRegistryConfig(
            require_approval_for_production=True,
            min_accuracy_for_production=0.85,
        )
        registry = ModelRegistry(config)
        
        # Register model with low accuracy
        model = registry.register_model(
            name="low-accuracy-model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/low/v1.0.0",
            metrics=ModelMetrics(accuracy=0.70),
            created_by="engineer@test.com",
        )
        
        # Try to promote to production - should fail
        with pytest.raises(ValueError) as exc_info:
            registry.request_promotion(
                model_id=model.model_id,
                to_stage=ModelStage.PRODUCTION,
                requester_id="engineer@test.com",
                justification="Need to deploy",
            )
        
        assert "accuracy" in str(exc_info.value).lower()


class TestModelRetrieval:
    """Test model retrieval functionality."""
    
    def test_get_model_by_id(self):
        """Test retrieving model by ID."""
        registry = ModelRegistry()
        
        model = registry.register_model(
            name="retrieval-test",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://test",
            created_by="test@test.com",
        )
        
        retrieved = registry.get_model(model.model_id)
        assert retrieved.model_id == model.model_id
        assert retrieved.name == "retrieval-test"
    
    def test_list_models_by_stage(self):
        """Test listing models by stage."""
        registry = ModelRegistry()
        
        # Register multiple models
        for i in range(3):
            registry.register_model(
                name=f"model-{i}",
                version="1.0.0",
                model_type=ModelType.CLASSIFICATION,
                artifact_uri=f"s3://models/m{i}",
                created_by="test@test.com",
            )
        
        # All should be in development
        dev_models = registry.list_models(stage=ModelStage.DEVELOPMENT)
        assert len(dev_models) == 3
    
    def test_get_latest_version(self):
        """Test getting latest version of a model."""
        registry = ModelRegistry()
        
        # Register multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            registry.register_model(
                name="versioned-model",
                version=version,
                model_type=ModelType.CLASSIFICATION,
                artifact_uri=f"s3://models/vm/{version}",
                created_by="test@test.com",
            )
        
        latest = registry.get_latest_model("versioned-model")
        assert latest.version == "2.0.0"


class TestModelCard:
    """Test model card generation."""
    
    def test_model_card_creation(self):
        """Test creating model with model card."""
        registry = ModelRegistry()
        
        model_card = ModelCard(
            model_name="Clinical Risk Predictor",
            model_version="1.0.0",
            model_type="Binary Classification",
            description="Predicts 30-day readmission risk",
            intended_use="Clinical decision support",
            out_of_scope_uses=["Primary diagnosis", "Treatment selection"],
            training_data_description="2 years of EHR data from 5 hospitals",
            evaluation_data_description="Held-out test set from 2023",
            ethical_considerations="Model performance varies by demographic",
            clinical_validation_status="Validated on 10,000 patient cohort",
            regulatory_status="For research use only",
        )
        
        model = registry.register_model(
            name="risk-predictor",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            artifact_uri="s3://models/risk/v1",
            model_card=model_card,
            created_by="ml_team@hospital.org",
        )
        
        assert model.model_card.intended_use == "Clinical decision support"
        assert "research" in model.model_card.regulatory_status.lower()
