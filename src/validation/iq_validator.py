"""
Installation Qualification (IQ) Validator

Validates that the ML environment is properly installed and configured
according to FDA 21 CFR Part 11 requirements.
"""

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pkg_resources


@dataclass
class IQCheckResult:
    """Result of a single IQ check."""
    
    check_name: str
    passed: bool
    expected: str
    actual: str
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IQReport:
    """Complete IQ validation report."""
    
    passed: bool
    checks: list[IQCheckResult]
    environment_info: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validator_version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "validator_version": self.validator_version,
            "environment_info": self.environment_info,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "expected": c.expected,
                    "actual": c.actual,
                    "details": c.details,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.checks
            ],
            "summary": {
                "total_checks": len(self.checks),
                "passed_checks": sum(1 for c in self.checks if c.passed),
                "failed_checks": sum(1 for c in self.checks if not c.passed),
            },
        }


class IQValidator:
    """
    Installation Qualification Validator for FDA 21 CFR Part 11 compliance.
    
    Validates:
    - Python version and environment
    - Required package versions
    - System dependencies
    - File permissions
    - Environment variables
    """
    
    REQUIRED_PACKAGES = {
        "numpy": ">=1.24.0",
        "pandas": ">=2.0.0",
        "scikit-learn": ">=1.3.0",
        "mlflow": ">=2.0.0",
    }
    
    MIN_PYTHON_VERSION = (3, 11)
    
    def __init__(
        self,
        required_packages: dict[str, str] | None = None,
        min_python_version: tuple[int, int] | None = None,
    ):
        """
        Initialize IQ validator.
        
        Args:
            required_packages: Dict of package names to version requirements.
            min_python_version: Minimum Python version as (major, minor).
        """
        self.required_packages = required_packages or self.REQUIRED_PACKAGES
        self.min_python_version = min_python_version or self.MIN_PYTHON_VERSION
        self.checks: list[IQCheckResult] = []
    
    def validate_environment(self) -> IQReport:
        """
        Run all IQ validation checks.
        
        Returns:
            IQReport with all check results.
        """
        self.checks = []
        
        # Run all validation checks
        self._check_python_version()
        self._check_required_packages()
        self._check_system_resources()
        self._check_environment_variables()
        self._check_file_permissions()
        
        # Determine overall pass/fail
        all_passed = all(c.passed for c in self.checks)
        
        return IQReport(
            passed=all_passed,
            checks=self.checks,
            environment_info=self._collect_environment_info(),
        )
    
    def _check_python_version(self) -> None:
        """Validate Python version meets requirements."""
        current = sys.version_info[:2]
        expected = f">={self.min_python_version[0]}.{self.min_python_version[1]}"
        actual = f"{current[0]}.{current[1]}"
        
        passed = current >= self.min_python_version
        
        self.checks.append(
            IQCheckResult(
                check_name="python_version",
                passed=passed,
                expected=expected,
                actual=actual,
                details=f"Full version: {sys.version}",
            )
        )
    
    def _check_required_packages(self) -> None:
        """Validate all required packages are installed with correct versions."""
        for package, version_req in self.required_packages.items():
            try:
                installed = pkg_resources.get_distribution(package)
                actual_version = installed.version
                
                # Check version requirement
                req = pkg_resources.Requirement.parse(f"{package}{version_req}")
                passed = installed in req
                
                self.checks.append(
                    IQCheckResult(
                        check_name=f"package_{package}",
                        passed=passed,
                        expected=version_req,
                        actual=actual_version,
                        details=f"Location: {installed.location}",
                    )
                )
            except pkg_resources.DistributionNotFound:
                self.checks.append(
                    IQCheckResult(
                        check_name=f"package_{package}",
                        passed=False,
                        expected=version_req,
                        actual="NOT INSTALLED",
                        details="Package not found in environment",
                    )
                )
    
    def _check_system_resources(self) -> None:
        """Validate system has sufficient resources."""
        import shutil
        
        # Check disk space (minimum 1GB free)
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        min_free_gb = 1.0
        
        self.checks.append(
            IQCheckResult(
                check_name="disk_space",
                passed=free_gb >= min_free_gb,
                expected=f">={min_free_gb}GB free",
                actual=f"{free_gb:.2f}GB free",
                details=f"Total: {total / (1024**3):.2f}GB",
            )
        )
    
    def _check_environment_variables(self) -> None:
        """Validate required environment variables are set."""
        import os
        
        # Check for common ML environment variables
        recommended_vars = ["MLFLOW_TRACKING_URI", "MODEL_REGISTRY_URI"]
        
        for var in recommended_vars:
            value = os.environ.get(var)
            self.checks.append(
                IQCheckResult(
                    check_name=f"env_{var}",
                    passed=value is not None,
                    expected="SET",
                    actual="SET" if value else "NOT SET",
                    details="Recommended for production deployment",
                )
            )
    
    def _check_file_permissions(self) -> None:
        """Validate file permissions for model artifacts directory."""
        import os
        import tempfile
        
        # Check we can write to temp directory
        temp_dir = Path(tempfile.gettempdir())
        can_write = os.access(temp_dir, os.W_OK)
        
        self.checks.append(
            IQCheckResult(
                check_name="temp_dir_writable",
                passed=can_write,
                expected="WRITABLE",
                actual="WRITABLE" if can_write else "NOT WRITABLE",
                details=f"Temp directory: {temp_dir}",
            )
        )
    
    def _collect_environment_info(self) -> dict[str, Any]:
        """Collect comprehensive environment information."""
        return {
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
            },
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": {
                dist.key: dist.version
                for dist in pkg_resources.working_set
            },
        }
    
    def generate_checksum(self, filepath: Path) -> str:
        """
        Generate SHA-256 checksum for a file.
        
        Used for validating model artifact integrity.
        
        Args:
            filepath: Path to file.
            
        Returns:
            Hex digest of SHA-256 hash.
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
