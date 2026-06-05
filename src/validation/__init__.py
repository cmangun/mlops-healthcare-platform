"""
FDA 21 CFR Part 11 Compliant Validation Framework

This module provides IQ/OQ/PQ validation protocols for ML models
in regulated healthcare environments.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "IQValidator",
    "OQValidator", 
    "PQValidator",
    "StatisticalValidator",
    "ValidationReportGenerator",
]

_EXPORT_MODULES = {
    "IQValidator": ".iq_validator",
    "OQValidator": ".oq_validator",
    "PQValidator": ".pq_validator",
    "StatisticalValidator": ".statistical",
    "ValidationReportGenerator": ".report_generator",
}


def __getattr__(name: str) -> Any:
    """Load each validation protocol without importing optional peers."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
