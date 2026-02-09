"""Policy helpers for benchmark checks."""
from __future__ import annotations


def _segmentation_policy_checks(
    *,
    segmented_shuffled_gain: float | None,
    y_global_for_check: float,
    segmented_data_gain: float | None,
    random_split_mean: float | None,
) -> tuple[bool, bool]:
    """Perform policy checks for segmentation benchmarks.

    Args:
        segmented_shuffled_gain: Gain from a model fit on shuffled data.
        y_global_for_check: Baseline global gain.
        segmented_data_gain: Gain from the data-segmented model.
        random_split_mean: Average gain from random splits.

    Returns:
        A tuple of (shuffled_ok, random_ok).
    """
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
    """Check if optimized oracle split outperforms the true oracle and data-based splits.

    Args:
        segmented_oracle_opt_gain: Gain from optimized oracle split.
        segmented_oracle_gain: Gain from true oracle split.
        segmented_data_honest_middle_gain: Honest gain from data-segmented split (middle).
        segmented_oracle_opt_honest_middle_gain: Honest gain from oracle-opt split (middle).
        segmented_data_honest_tail_gain: Honest gain from data-segmented split (tail).
        segmented_oracle_opt_honest_tail_gain: Honest gain from oracle-opt split (tail).
        margin: Allowed slack for comparisons.

    Returns:
        A tuple of (opt_ge_true, data_le_opt).
    """
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
    """Check if the global honest gain on the tail is non-negative.

    Args:
        y_global_honest_tail_gain: The computed gain value.

    Returns:
        True if the gain is >= 0 or None.
    """
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
    """Perform margin-based checks for MNJ gating performance.

    Args:
        mnj_gain: Gain from MNJ-based segmentation.
        dy_gain: Gain from dY/dt-based segmentation.
        random_split_mean: Average gain from random splits.
        mnj_shuffled_gain: Gain from MNJ on shuffled data.
        margin: Minimum required performance difference.

    Returns:
        A tuple of (mnj_beats_dy, mnj_beats_random, mnj_shuffled_collapse).
    """
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
