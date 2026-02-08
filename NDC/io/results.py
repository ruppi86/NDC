"""Result serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import asdict, dataclass


def save_results(path: str | Path, results: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


@dataclass
class ComparisonResult:
    baseline_config: str
    counterfactual_config: str
    baseline_metrics: dict[str, float]
    counterfactual_metrics: dict[str, float]
    deltas: dict[str, float]
    effect_sizes: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_comparison(path: str | Path, result: ComparisonResult) -> None:
    save_results(path, result.to_dict())

