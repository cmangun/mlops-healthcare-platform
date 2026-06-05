"""
MLOps Healthcare Platform

Enterprise MLOps platform for healthcare with FDA 21 CFR Part 11 compliance,
automated model validation, and complete audit trails.
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    # Validation
    "IQValidator",
    "OQValidator",
    "PQValidator",
    "StatisticalValidator",
    "ValidationReportGenerator",
    # Training
    "HealthcareTrainer",
    "HyperparameterOptimizer",
]

_EXPORT_MODULES = {
    "IQValidator": ".validation",
    "OQValidator": ".validation",
    "PQValidator": ".validation",
    "StatisticalValidator": ".validation",
    "ValidationReportGenerator": ".validation",
    "HealthcareTrainer": ".training",
    "HyperparameterOptimizer": ".training",
}


def __getattr__(name: str) -> Any:
    """Load optional platform components only when they are requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
