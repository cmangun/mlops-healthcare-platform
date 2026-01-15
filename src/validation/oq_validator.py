"""
Operational Qualification (OQ) Validator

Validates that ML models function correctly according to specifications
as required by FDA 21 CFR Part 11.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np


class MLModel(Protocol):
    """Protocol for ML models that can be validated."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        ...


@dataclass
class OQTestCase:
    """A single OQ test case."""
    
    name: str
    description: str
    input_data: np.ndarray
    expected_output: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    tolerance: float = 1e-6
    max_latency_ms: float = 1000.0


@dataclass
class OQTestResult:
    """Result of a single OQ test."""
    
    test_name: str
    passed: bool
    details: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OQReport:
    """Complete OQ validation report."""
    
    passed: bool
    tests: list[OQTestResult]
    model_info: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validator_version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "validator_version": self.validator_version,
            "model_info": self.model_info,
            "tests": [
                {
                    "test_name": t.test_name,
                    "passed": t.passed,
                    "details": t.details,
                    "latency_ms": t.latency_ms,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in self.tests
            ],
            "summary": {
                "total_tests": len(self.tests),
                "passed_tests": sum(1 for t in self.tests if t.passed),
                "failed_tests": sum(1 for t in self.tests if not t.passed),
                "avg_latency_ms": np.mean([t.latency_ms for t in self.tests]),
            },
        }


class OQValidator:
    """
    Operational Qualification Validator for FDA 21 CFR Part 11 compliance.
    
    Validates:
    - Model prediction functionality
    - Output shape and type correctness
    - Deterministic behavior
    - Latency requirements
    - Edge case handling
    """
    
    def __init__(self, model: Any):
        """
        Initialize OQ validator.
        
        Args:
            model: ML model to validate.
        """
        self.model = model
        self.test_results: list[OQTestResult] = []
    
    def validate_functionality(
        self,
        test_cases: list[OQTestCase] | None = None,
    ) -> OQReport:
        """
        Run all OQ validation tests.
        
        Args:
            test_cases: Custom test cases. If None, uses default tests.
            
        Returns:
            OQReport with all test results.
        """
        self.test_results = []
        
        if test_cases is None:
            test_cases = self._generate_default_test_cases()
        
        for test_case in test_cases:
            self._run_test_case(test_case)
        
        # Run additional standard tests
        self._test_determinism()
        self._test_empty_input_handling()
        self._test_large_batch_handling()
        
        all_passed = all(t.passed for t in self.test_results)
        
        return OQReport(
            passed=all_passed,
            tests=self.test_results,
            model_info=self._collect_model_info(),
        )
    
    def _run_test_case(self, test_case: OQTestCase) -> None:
        """Run a single test case."""
        try:
            # Measure latency
            start_time = time.perf_counter()
            output = self.model.predict(test_case.input_data)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Validate output
            passed = True
            details = []
            
            # Check shape
            if test_case.expected_shape is not None:
                if output.shape != test_case.expected_shape:
                    passed = False
                    details.append(
                        f"Shape mismatch: expected {test_case.expected_shape}, "
                        f"got {output.shape}"
                    )
            
            # Check values
            if test_case.expected_output is not None:
                if not np.allclose(
                    output, test_case.expected_output, atol=test_case.tolerance
                ):
                    passed = False
                    details.append(
                        f"Output mismatch (tolerance={test_case.tolerance})"
                    )
            
            # Check latency
            if latency_ms > test_case.max_latency_ms:
                passed = False
                details.append(
                    f"Latency exceeded: {latency_ms:.2f}ms > "
                    f"{test_case.max_latency_ms}ms"
                )
            
            self.test_results.append(
                OQTestResult(
                    test_name=test_case.name,
                    passed=passed,
                    details="; ".join(details) if details else "All checks passed",
                    latency_ms=latency_ms,
                )
            )
            
        except Exception as e:
            self.test_results.append(
                OQTestResult(
                    test_name=test_case.name,
                    passed=False,
                    details=f"Exception: {str(e)}",
                    latency_ms=0.0,
                )
            )
    
    def _test_determinism(self) -> None:
        """Test that model produces deterministic results."""
        try:
            # Generate random input
            np.random.seed(42)
            test_input = np.random.randn(10, 5)
            
            # Run multiple times
            results = []
            for _ in range(3):
                output = self.model.predict(test_input)
                results.append(output)
            
            # Check all results are identical
            is_deterministic = all(
                np.array_equal(results[0], r) for r in results[1:]
            )
            
            self.test_results.append(
                OQTestResult(
                    test_name="determinism",
                    passed=is_deterministic,
                    details="Model produces consistent results" if is_deterministic
                    else "Model produces non-deterministic results",
                    latency_ms=0.0,
                )
            )
            
        except Exception as e:
            self.test_results.append(
                OQTestResult(
                    test_name="determinism",
                    passed=False,
                    details=f"Exception: {str(e)}",
                    latency_ms=0.0,
                )
            )
    
    def _test_empty_input_handling(self) -> None:
        """Test model handles empty input gracefully."""
        try:
            empty_input = np.array([]).reshape(0, 5)
            output = self.model.predict(empty_input)
            
            # Should return empty array without error
            passed = output.shape[0] == 0
            
            self.test_results.append(
                OQTestResult(
                    test_name="empty_input_handling",
                    passed=passed,
                    details="Model handles empty input correctly" if passed
                    else f"Unexpected output shape: {output.shape}",
                    latency_ms=0.0,
                )
            )
            
        except Exception as e:
            # Some models may legitimately raise errors on empty input
            self.test_results.append(
                OQTestResult(
                    test_name="empty_input_handling",
                    passed=True,  # Graceful error is acceptable
                    details=f"Model raises error on empty input: {str(e)}",
                    latency_ms=0.0,
                )
            )
    
    def _test_large_batch_handling(self) -> None:
        """Test model handles large batches."""
        try:
            large_input = np.random.randn(10000, 5)
            
            start_time = time.perf_counter()
            output = self.model.predict(large_input)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            passed = output.shape[0] == 10000 and latency_ms < 10000
            
            self.test_results.append(
                OQTestResult(
                    test_name="large_batch_handling",
                    passed=passed,
                    details=f"Processed 10K samples in {latency_ms:.2f}ms",
                    latency_ms=latency_ms,
                )
            )
            
        except Exception as e:
            self.test_results.append(
                OQTestResult(
                    test_name="large_batch_handling",
                    passed=False,
                    details=f"Exception: {str(e)}",
                    latency_ms=0.0,
                )
            )
    
    def _generate_default_test_cases(self) -> list[OQTestCase]:
        """Generate default test cases based on model inspection."""
        return [
            OQTestCase(
                name="basic_prediction",
                description="Basic prediction with standard input",
                input_data=np.random.randn(10, 5),
                expected_shape=(10,),
            ),
            OQTestCase(
                name="single_sample",
                description="Single sample prediction",
                input_data=np.random.randn(1, 5),
                expected_shape=(1,),
            ),
        ]
    
    def _collect_model_info(self) -> dict[str, Any]:
        """Collect model metadata."""
        info = {
            "model_type": type(self.model).__name__,
            "model_module": type(self.model).__module__,
        }
        
        # Try to get additional attributes
        for attr in ["n_features_in_", "n_classes_", "classes_"]:
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                info[attr] = value
        
        return info
