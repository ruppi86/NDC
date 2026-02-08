from pathlib import Path

from ndc_analysis.guardrails import (
    find_forbidden_imports,
    find_non_allowlisted_imports,
    iter_python_files,
)


def test_ndc_analysis_has_no_simulator_imports():
    root = Path(__file__).resolve().parents[1] / "ndc_analysis"
    violations = find_forbidden_imports(iter_python_files(root))
    assert violations == [], f"Forbidden simulator imports found: {violations}"


def test_ndc_analysis_uses_allowlisted_imports():
    root = Path(__file__).resolve().parents[1] / "ndc_analysis"
    violations = find_non_allowlisted_imports(iter_python_files(root))
    assert violations == [], f"Non-allowlisted imports found: {violations}"
