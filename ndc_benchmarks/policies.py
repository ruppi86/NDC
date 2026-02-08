"""Policy helpers for benchmark checks."""
from __future__ import annotations


def _segmentation_policy_checks(
    *,
    segmented_shuffled_gain: float | None,
    y_global_for_check: float,
    segmented_data_gain: float | None,
    random_split_mean: float | None,
) -> tuple[bool, bool]:
    segmented_shuffled_ok = True
    if segmented_shuffled_gain is not None:
        upper_bound = max(y_global_for_check, 0.0)
        segmented_shuffled_ok = segmented_shuffled_gain <= upper_bound + 1e-12
    segmented_random_ok = True
    if segmented_data_gain is not None and random_split_mean is not None:
        segmented_random_ok = segmented_data_gain >= random_split_mean
    return segmented_shuffled_ok, segmented_random_ok


def _oracle_opt_policy_checks(
    *,
    segmented_oracle_opt_gain: float | None,
    segmented_oracle_gain: float | None,
    segmented_data_honest_middle_gain: float | None,
    segmented_oracle_opt_honest_middle_gain: float | None,
    segmented_data_honest_tail_gain: float | None,
    segmented_oracle_opt_honest_tail_gain: float | None,
    margin: float,
) -> tuple[bool, bool]:
    oracle_opt_ge_oracle_true = True
    if segmented_oracle_opt_gain is not None and segmented_oracle_gain is not None:
        oracle_opt_ge_oracle_true = (
            segmented_oracle_opt_gain >= segmented_oracle_gain - 1e-12
        )

    data_honest_le_oracle_opt = True
    if (
        segmented_data_honest_middle_gain is not None
        and segmented_oracle_opt_honest_middle_gain is not None
    ):
        data_honest_le_oracle_opt = (
            segmented_data_honest_middle_gain
            <= segmented_oracle_opt_honest_middle_gain + margin
        )
    elif (
        segmented_data_honest_tail_gain is not None
        and segmented_oracle_opt_honest_tail_gain is not None
    ):
        data_honest_le_oracle_opt = (
            segmented_data_honest_tail_gain
            <= segmented_oracle_opt_honest_tail_gain + margin
        )
    return oracle_opt_ge_oracle_true, data_honest_le_oracle_opt


def _global_honest_tail_nonnegative(y_global_honest_tail_gain: float | None) -> bool:
    if y_global_honest_tail_gain is None:
        return True
    return y_global_honest_tail_gain >= 0.0


def _gate_margin_checks(
    *,
    mnj_gain: float | None,
    dy_gain: float | None,
    random_split_mean: float | None,
    mnj_shuffled_gain: float | None,
    margin: float,
) -> tuple[bool, bool, bool]:
    mnj_gate_beats_dy_gate = True
    if mnj_gain is not None and dy_gain is not None:
        mnj_gate_beats_dy_gate = mnj_gain > dy_gain + margin
    mnj_gate_beats_random = True
    if mnj_gain is not None and random_split_mean is not None:
        mnj_gate_beats_random = mnj_gain > random_split_mean + margin
    mnj_gate_shuffled_collapse = True
    if mnj_shuffled_gain is not None and random_split_mean is not None:
        mnj_gate_shuffled_collapse = mnj_shuffled_gain <= random_split_mean + 1e-12
    return mnj_gate_beats_dy_gate, mnj_gate_beats_random, mnj_gate_shuffled_collapse
