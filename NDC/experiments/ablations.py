"""Ablation preset helpers and comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from NDC.config.schema import DriverConfig, ExperimentConfig, RVCConfig
from NDC.experiments.runner import run_experiment_from_config
from NDC.io.results import ComparisonResult, save_comparison


PRESETS = {
    "baseline": "baseline.yaml",
    "gate_ablation": "gate_ablation.yaml",
    "timing_counterfactual": "timing_counterfactual.yaml",
    "resolution_stress": "resolution_stress.yaml",
    "landscape_swap": "landscape_swap.yaml",
}


def preset_path(name: str) -> Path:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'")
    return Path(__file__).resolve().parents[1] / "config" / "presets" / PRESETS[name]


def _aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_list[0].keys() if metric_list else []
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


def _std(values: np.ndarray) -> float:
    if values.size < 2:
        return float(np.std(values, ddof=0))
    return float(np.std(values, ddof=1))


def _compute_effect_sizes(
    baseline: list[dict[str, float]], counterfactual: list[dict[str, float]]
) -> dict[str, float]:
    keys = baseline[0].keys() if baseline else []
    effect_sizes: dict[str, float] = {}
    for key in keys:
        b_vals = np.array([m[key] for m in baseline], dtype=float)
        c_vals = np.array([m[key] for m in counterfactual], dtype=float)
        pooled = np.sqrt((_std(b_vals) ** 2 + _std(c_vals) ** 2) / 2.0)
        if pooled == 0.0 or np.isnan(pooled):
            effect_sizes[key] = 0.0
        else:
            effect_sizes[key] = float((np.mean(c_vals) - np.mean(b_vals)) / pooled)
    return effect_sizes


def run_ablation_comparison(
    baseline_config: ExperimentConfig,
    counterfactual_config: ExperimentConfig,
    output_dir: Path,
    n_runs: int = 5,
) -> ComparisonResult:
    baseline_metrics: list[dict[str, float]] = []
    counterfactual_metrics: list[dict[str, float]] = []

    for i in range(n_runs):
        b_cfg = baseline_config.model_copy(deep=True)
        c_cfg = counterfactual_config.model_copy(deep=True)
        b_cfg.seed = baseline_config.seed + i
        c_cfg.seed = counterfactual_config.seed + i

        b_result = run_experiment_from_config(b_cfg, output_dir / f"baseline_{i}")
        c_result = run_experiment_from_config(c_cfg, output_dir / f"counterfactual_{i}")

        baseline_metrics.append(b_result["oracle_metrics"])
        counterfactual_metrics.append(c_result["oracle_metrics"])

    baseline_means = _aggregate_metrics(baseline_metrics)
    counterfactual_means = _aggregate_metrics(counterfactual_metrics)
    deltas = {k: counterfactual_means[k] - baseline_means[k] for k in baseline_means}
    effect_sizes = _compute_effect_sizes(baseline_metrics, counterfactual_metrics)

    result = ComparisonResult(
        baseline_config=baseline_config.model_dump_json(),
        counterfactual_config=counterfactual_config.model_dump_json(),
        baseline_metrics=baseline_means,
        counterfactual_metrics=counterfactual_means,
        deltas=deltas,
        effect_sizes=effect_sizes,
    )

    save_comparison(output_dir / "comparison.json", result)
    return result


def run_gate_ablation(base_config: ExperimentConfig, output_dir: Path) -> ComparisonResult:
    counterfactual = base_config.model_copy(deep=True)
    counterfactual.rvc = RVCConfig(name="null_gate", params={"sigma0": 0.6})
    return run_ablation_comparison(base_config, counterfactual, output_dir)


def run_timing_counterfactual(base_config: ExperimentConfig, output_dir: Path) -> ComparisonResult:
    counterfactual = base_config.model_copy(deep=True)
    counterfactual.driver = DriverConfig(
        name="phase_randomized_rhythm",
        params={
            "base_driver": base_config.driver.model_dump(),
            "t_start": base_config.time.t_start,
            "t_end": base_config.time.t_end,
            "dt": base_config.time.dt_sim,
        },
    )
    return run_ablation_comparison(base_config, counterfactual, output_dir)

