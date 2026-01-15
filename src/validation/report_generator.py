"""
FDA Validation Report Generator

Generates compliant validation documentation for FDA submissions.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .iq_validator import IQReport
from .oq_validator import OQReport
from .pq_validator import PQReport


@dataclass
class ValidationPackage:
    """Complete FDA validation package."""
    
    model_id: str
    model_version: str
    intended_use: str
    iq_report: IQReport
    oq_report: OQReport
    pq_report: PQReport
    timestamp: datetime
    validator_version: str = "1.0.0"
    
    @property
    def overall_passed(self) -> bool:
        """Check if all validations passed."""
        return (
            self.iq_report.passed and
            self.oq_report.passed and
            self.pq_report.passed
        )


class ValidationReportGenerator:
    """
    Generates FDA 21 CFR Part 11 compliant validation reports.
    
    Produces:
    - IQ/OQ/PQ summary reports
    - Detailed test results
    - Audit trail documentation
    - Executive summary
    """
    
    def __init__(
        self,
        model_id: str,
        model_version: str,
        intended_use: str,
    ):
        """
        Initialize report generator.
        
        Args:
            model_id: Unique model identifier.
            model_version: Model version string.
            intended_use: Intended use statement for regulatory.
        """
        self.model_id = model_id
        self.model_version = model_version
        self.intended_use = intended_use
    
    def create_validation_package(
        self,
        iq_report: IQReport,
        oq_report: OQReport,
        pq_report: PQReport,
    ) -> ValidationPackage:
        """
        Create complete validation package.
        
        Args:
            iq_report: Installation Qualification report.
            oq_report: Operational Qualification report.
            pq_report: Performance Qualification report.
            
        Returns:
            Complete ValidationPackage.
        """
        return ValidationPackage(
            model_id=self.model_id,
            model_version=self.model_version,
            intended_use=self.intended_use,
            iq_report=iq_report,
            oq_report=oq_report,
            pq_report=pq_report,
            timestamp=datetime.utcnow(),
        )
    
    def generate_executive_summary(
        self,
        package: ValidationPackage,
    ) -> str:
        """
        Generate executive summary for FDA submission.
        
        Args:
            package: Complete validation package.
            
        Returns:
            Markdown formatted executive summary.
        """
        status = "PASSED" if package.overall_passed else "FAILED"
        
        summary = f"""# Validation Executive Summary

## Model Information
- **Model ID:** {package.model_id}
- **Version:** {package.model_version}
- **Validation Date:** {package.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
- **Overall Status:** {status}

## Intended Use
{package.intended_use}

## Validation Summary

### Installation Qualification (IQ)
- **Status:** {"PASSED" if package.iq_report.passed else "FAILED"}
- **Checks Performed:** {len(package.iq_report.checks)}
- **Checks Passed:** {sum(1 for c in package.iq_report.checks if c.passed)}

### Operational Qualification (OQ)
- **Status:** {"PASSED" if package.oq_report.passed else "FAILED"}
- **Tests Performed:** {len(package.oq_report.tests)}
- **Tests Passed:** {sum(1 for t in package.oq_report.tests if t.passed)}

### Performance Qualification (PQ)
- **Status:** {"PASSED" if package.pq_report.passed else "FAILED"}
- **Metrics Evaluated:** {len(package.pq_report.metrics)}
- **Metrics Passed:** {sum(1 for m in package.pq_report.metrics if m.passed)}

## Key Performance Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
"""
        
        for metric in package.pq_report.metrics:
            status_emoji = "✓" if metric.passed else "✗"
            summary += f"| {metric.metric_name} | {metric.value:.4f} | {metric.threshold:.4f} | {status_emoji} |\n"
        
        summary += """
## Conclusion

"""
        if package.overall_passed:
            summary += """The model has successfully completed all validation phases (IQ, OQ, PQ) 
and meets the requirements for deployment in a regulated healthcare environment 
in accordance with FDA 21 CFR Part 11 guidelines."""
        else:
            summary += """The model has NOT passed all validation requirements. 
Review the detailed reports below to identify and address failures before deployment."""
        
        return summary
    
    def generate_iq_report(self, iq_report: IQReport) -> str:
        """Generate detailed IQ report."""
        report = f"""# Installation Qualification Report

**Generated:** {iq_report.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
**Validator Version:** {iq_report.validator_version}
**Status:** {"PASSED" if iq_report.passed else "FAILED"}

## Environment Information

### Python
- Version: {iq_report.environment_info.get('python', {}).get('version', 'N/A')}
- Executable: {iq_report.environment_info.get('python', {}).get('executable', 'N/A')}

### System
- OS: {iq_report.environment_info.get('system', {}).get('os', 'N/A')}
- Machine: {iq_report.environment_info.get('system', {}).get('machine', 'N/A')}

## Validation Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""
        
        for check in iq_report.checks:
            status = "✓ PASS" if check.passed else "✗ FAIL"
            report += f"| {check.check_name} | {check.expected} | {check.actual} | {status} |\n"
        
        return report
    
    def generate_oq_report(self, oq_report: OQReport) -> str:
        """Generate detailed OQ report."""
        report = f"""# Operational Qualification Report

**Generated:** {oq_report.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
**Validator Version:** {oq_report.validator_version}
**Status:** {"PASSED" if oq_report.passed else "FAILED"}

## Model Information
- Type: {oq_report.model_info.get('model_type', 'N/A')}
- Module: {oq_report.model_info.get('model_module', 'N/A')}

## Functional Tests

| Test | Latency (ms) | Status | Details |
|------|--------------|--------|---------|
"""
        
        for test in oq_report.tests:
            status = "✓ PASS" if test.passed else "✗ FAIL"
            report += f"| {test.test_name} | {test.latency_ms:.2f} | {status} | {test.details} |\n"
        
        return report
    
    def generate_pq_report(self, pq_report: PQReport) -> str:
        """Generate detailed PQ report."""
        report = f"""# Performance Qualification Report

**Generated:** {pq_report.timestamp.strftime("%Y-%m-%d %H:%M UTC")}
**Validator Version:** {pq_report.validator_version}
**Status:** {"PASSED" if pq_report.passed else "FAILED"}

## Performance Metrics

| Metric | Value | Threshold | CI (95%) | Status |
|--------|-------|-----------|----------|--------|
"""
        
        for metric in pq_report.metrics:
            status = "✓ PASS" if metric.passed else "✗ FAIL"
            ci_str = f"[{metric.confidence_interval[0]:.3f}, {metric.confidence_interval[1]:.3f}]" \
                if metric.confidence_interval else "N/A"
            report += f"| {metric.metric_name} | {metric.value:.4f} | {metric.threshold:.4f} | {ci_str} | {status} |\n"
        
        report += f"""
## Confusion Matrix

```
{pq_report.confusion_matrix}
```

## Classification Report

```
{pq_report.classification_report}
```
"""
        
        return report
    
    def save_package(
        self,
        package: ValidationPackage,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """
        Save validation package to files.
        
        Args:
            package: Validation package to save.
            output_dir: Output directory.
            formats: List of formats ("md", "json").
            
        Returns:
            Dict mapping format to output path.
        """
        if formats is None:
            formats = ["md", "json"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        timestamp = package.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if "md" in formats:
            # Generate combined markdown report
            md_content = self.generate_executive_summary(package)
            md_content += "\n\n---\n\n"
            md_content += self.generate_iq_report(package.iq_report)
            md_content += "\n\n---\n\n"
            md_content += self.generate_oq_report(package.oq_report)
            md_content += "\n\n---\n\n"
            md_content += self.generate_pq_report(package.pq_report)
            
            md_path = output_dir / f"validation_report_{timestamp}.md"
            md_path.write_text(md_content)
            outputs["md"] = md_path
        
        if "json" in formats:
            # Generate JSON for programmatic access
            json_data = {
                "model_id": package.model_id,
                "model_version": package.model_version,
                "intended_use": package.intended_use,
                "timestamp": package.timestamp.isoformat(),
                "overall_passed": package.overall_passed,
                "iq_report": package.iq_report.to_dict(),
                "oq_report": package.oq_report.to_dict(),
                "pq_report": package.pq_report.to_dict(),
            }
            
            json_path = output_dir / f"validation_report_{timestamp}.json"
            json_path.write_text(json.dumps(json_data, indent=2))
            outputs["json"] = json_path
        
        return outputs
