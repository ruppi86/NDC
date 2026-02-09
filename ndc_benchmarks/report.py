"""Report rendering for benchmarks."""
from __future__ import annotations

from typing import Any

from .types import BenchmarkResult, REPORT_VERSION


def _render_report(results: dict[str, BenchmarkResult], claims: list[dict[str, Any]]) -> str:
    """Render a Markdown report from benchmark results and claims.

    Args:
        results: Dictionary mapping benchmark IDs to their results.
        claims: List of claim dictionaries from the claims matrix.

    Returns:
        A string containing the formatted Markdown report.
    """
    lines = ["# Benchmark Report", f"report_version: {REPORT_VERSION}", ""]
    if claims:
        lines.append("## Claims Matrix")
        lines.append("| claim | test | negative_control | expected_failure_mode |")
        lines.append("| --- | --- | --- | --- |")
        for item in claims:
            lines.append(
                f"| {item.get('claim')} | {item.get('test')} | "
                f"{item.get('negative_control')} | {item.get('expected_failure_mode')} |"
            )
        lines.append("")
    for bid, result in results.items():
        lines.append(f"## {bid}")
        lines.append(f"status: **{result.status}**")
        lines.append("")
        if "golden_a" in bid:
            lines.append("### Golden A1: Within-regime segmented prediction (primary)")
            lines.append(
                "Interpretation: compares data-segmented vs random split under "
                "mixed-regime eval (honest_middle)."
            )
            lines.append("")
            lines.append("### Golden A2: Cross-regime tail generalization (stress test)")
            lines.append(
                "Interpretation: tail eval can go negative; treat as WARN-level "
                "generalization stress."
            )
            lines.append("")
        lines.append("metrics:")
        for key, value in result.metrics.items():
            lines.append(f"- {key}: {value}")
        if "dt_obs_sweep" in result.metrics and "dt_obs_effects" in result.metrics:
            lines.append("")
            lines.append("dt_obs_curve:")
            lines.append("| dt_obs | score |")
            lines.append("| --- | --- |")
            for dt, score in zip(
                result.metrics["dt_obs_sweep"], result.metrics["dt_obs_effects"]
            ):
                lines.append(f"| {dt} | {round(float(score), 6)} |")
        if "segmentation_selection_fraction" in result.metrics:
            lines.append("")
            lines.append("honest_middle:")
            lines.append("| model | gain | mse_model | mse_baseline |")
            lines.append("| --- | --- | --- | --- |")
            middle_rows = [
                (
                    "global",
                    result.metrics.get("prediction_gain_y_global_honest_middle"),
                    None,
                    None,
                ),
                (
                    "oracle_true",
                    result.metrics.get(
                        "prediction_gain_oracle_true_segmented_global_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "oracle_opt",
                    result.metrics.get(
                        "prediction_gain_oracle_opt_segmented_global_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "data_segmented",
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle_mse_model"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle_mse_baseline"
                    ),
                ),
                (
                    "segmented_shuffled",
                    result.metrics.get("prediction_gain_segmented_shuffled_honest_middle"),
                    None,
                    None,
                ),
                (
                    "random_split_mean",
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_mean"
                    ),
                    None,
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_std"
                    ),
                ),
                (
                    "dy_gate",
                    result.metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "mnj_gate",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "mnj_gate_shuffled",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle"
                    ),
                    None,
                    None,
                ),
            ]
            for label, gain, mse_model, mse_baseline in middle_rows:
                lines.append(
                    f"| {label} | {gain} | {mse_model} | {mse_baseline} |"
                )
            lines.append("")
            lines.append(
                f"honest_middle_split: selection_fraction="
                f"{result.metrics.get('segmentation_selection_fraction')}, "
                f"fit_T={result.metrics.get('segmentation_fit_T')}, "
                f"eval_start={result.metrics.get('segmentation_eval_middle_start')}, "
                f"eval_end={result.metrics.get('segmentation_eval_middle_end')}"
            )

            lines.append("")
            lines.append("honest_tail:")
            lines.append("| model | gain | mse_model | mse_baseline |")
            lines.append("| --- | --- | --- | --- |")
            tail_rows = [
                (
                    "global",
                    result.metrics.get("prediction_gain_y_global_honest_tail"),
                    None,
                    None,
                ),
                (
                    "oracle_true",
                    result.metrics.get(
                        "prediction_gain_oracle_true_segmented_global_honest_tail"
                    ),
                    None,
                    None,
                ),
                (
                    "oracle_opt",
                    result.metrics.get(
                        "prediction_gain_oracle_opt_segmented_global_honest_tail"
                    ),
                    None,
                    None,
                ),
                (
                    "data_segmented",
                    result.metrics.get("prediction_gain_data_segmented_global_honest_tail"),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_mse_model"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_mse_baseline"
                    ),
                ),
                (
                    "segmented_shuffled",
                    result.metrics.get("prediction_gain_segmented_shuffled_honest_tail"),
                    None,
                    None,
                ),
                (
                    "random_split_mean",
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_mean"
                    ),
                    None,
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_std"
                    ),
                ),
                (
                    "dy_gate",
                    result.metrics.get("prediction_gain_segmented_dy_gate_honest_tail"),
                    None,
                    None,
                ),
                (
                    "mnj_gate",
                    result.metrics.get("prediction_gain_segmented_mnj_gate_honest_tail"),
                    None,
                    None,
                ),
                (
                    "mnj_gate_shuffled",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_tail"
                    ),
                    None,
                    None,
                ),
            ]
            for label, gain, mse_model, mse_baseline in tail_rows:
                lines.append(
                    f"| {label} | {gain} | {mse_model} | {mse_baseline} |"
                )
            lines.append("")
            lines.append(
                f"honest_tail_split: selection_fraction="
                f"{result.metrics.get('segmentation_selection_fraction')}, "
                f"fit_T={result.metrics.get('segmentation_fit_T')}, "
                f"eval_start={result.metrics.get('segmentation_eval_start')}, "
                f"eval_T={result.metrics.get('segmentation_eval_T')}"
            )
        lines.append("")
        lines.append("checks:")
        for key, value in result.checks.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _build_sections(results: dict[str, BenchmarkResult]) -> list[dict[str, Any]]:
    """Build structured report sections for JSON serialization.

    Args:
        results: Dictionary mapping benchmark IDs to their results.

    Returns:
        A list of section dictionaries containing title, metrics, and checks.
    """
    sections: list[dict[str, Any]] = []
    a_key = "golden_a_piecewise"
    if a_key in results:
        metrics = results[a_key].metrics
        checks = results[a_key].checks
        sections.append(
            {
                "id": "golden_a1",
                "title": "Golden A1: Within-regime segmented prediction (primary)",
                "metrics": {
                    "golden_a1.gain_segmented_mnj_gate_honest_middle": metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_middle"
                    ),
                    "golden_a1.gain_segmented_dy_gate_honest_middle": metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_middle"
                    ),
                    "golden_a1.gain_segmented_random_split_honest_middle_mean": metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_mean"
                    ),
                    "golden_a1.gain_segmented_random_split_honest_middle_std": metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_std"
                    ),
                    "golden_a1.gain_segmented_mnj_gate_shuffled_honest_middle": metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle"
                    ),
                    "golden_a1.mnj_gate_trust_coverage": metrics.get(
                        "mnj_gate_trust_coverage"
                    ),
                    "golden_a1.mnj_gate_trust_score_p50": metrics.get(
                        "mnj_gate_trust_score_p50"
                    ),
                    "golden_a1.mnj_gate_fail_residual_fraction": metrics.get(
                        "mnj_gate_fail_residual_fraction"
                    ),
                    "golden_a1.mnj_gate_split_idx_fit": metrics.get(
                        "prediction_gain_segmented_mnj_gate_split_idx_fit"
                    ),
                    "golden_a1.dy_gate_split_idx_fit": metrics.get(
                        "prediction_gain_segmented_dy_gate_split_idx_fit"
                    ),
                },
                "checks": {
                    "golden_a1.mnj_gate_beats_dy_gate": checks.get(
                        "mnj_gate_beats_dy_gate"
                    ),
                    "golden_a1.mnj_gate_beats_random": checks.get(
                        "mnj_gate_beats_random"
                    ),
                    "golden_a1.mnj_gate_shuffled_collapse": checks.get(
                        "mnj_gate_shuffled_collapse"
                    ),
                },
            }
        )
        sections.append(
            {
                "id": "golden_a2",
                "title": "Golden A2: Cross-regime tail generalization (stress test)",
                "metrics": {
                    "golden_a2.gain_y_global_honest_tail": metrics.get(
                        "prediction_gain_y_global_honest_tail"
                    ),
                    "golden_a2.gain_segmented_mnj_gate_honest_tail": metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_tail"
                    ),
                    "golden_a2.gain_segmented_dy_gate_honest_tail": metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_tail"
                    ),
                    "golden_a2.eval_tail_pre_fraction": metrics.get(
                        "segmentation_eval_tail_pre_fraction"
                    ),
                    "golden_a2.eval_tail_post_fraction": metrics.get(
                        "segmentation_eval_tail_post_fraction"
                    ),
                },
                "checks": {
                    "golden_a2.global_honest_tail_nonnegative": checks.get(
                        "global_honest_tail_nonnegative"
                    )
                },
            }
        )
    return sections
