"""Tests for installation qualification checks."""

from src.validation.iq_validator import IQValidator


def test_required_package_check_accepts_installed_version():
    validator = IQValidator(required_packages={"packaging": ">=23.0"})

    validator._check_required_packages()

    assert len(validator.checks) == 1
    assert validator.checks[0].check_name == "package_packaging"
    assert validator.checks[0].passed is True
    assert validator.checks[0].actual != "NOT INSTALLED"


def test_required_package_check_reports_missing_distribution():
    validator = IQValidator(
        required_packages={"package-that-does-not-exist-codex": ">=1.0"}
    )

    validator._check_required_packages()

    assert len(validator.checks) == 1
    assert validator.checks[0].passed is False
    assert validator.checks[0].actual == "NOT INSTALLED"
