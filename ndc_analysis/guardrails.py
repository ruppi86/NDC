"""Guardrails to enforce analysis/simulator separation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable


FORBIDDEN_PREFIXES = ("NDC", "ndc_sim")
ALLOWLIST_MODULES = {"ndc_analysis", "numpy", "h5py"}


def _stdlib_modules() -> set[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return set(names)
    return set()


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            yield path
        else:
            yield path


def find_forbidden_imports(
    paths: Iterable[Path], forbidden_prefixes: tuple[str, ...] = FORBIDDEN_PREFIXES
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(forbidden_prefixes):
                    violations.append(f"{path}: from {node.module} import ...")
    return violations


def find_non_allowlisted_imports(
    paths: Iterable[Path], allowlist: set[str] | None = None
) -> list[str]:
    allowlist = set(allowlist or set())
    allowlist = allowlist.union(ALLOWLIST_MODULES, _stdlib_modules())
    violations: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowlist:
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in allowlist:
                        violations.append(f"{path}: from {node.module} import ...")
    return violations
