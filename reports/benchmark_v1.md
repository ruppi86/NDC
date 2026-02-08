# Benchmark Report
report_version: 0.1

## Claims Matrix
| claim | test | negative_control | expected_failure_mode |
| --- | --- | --- | --- |
| MNJ detects local dynamical regime switches | golden_a_piecewise | time_shuffle, random_neighbors | effect size collapses under controls |
| Regime-aware modeling improves one-step prediction vs a single global model under mixed regimes | prediction_gain_data_segmented_global_honest_middle vs random_split_honest_middle_mean | segmented_shuffled, random_split_mean | data-segmented ≈ random or controls do not collapse |
| Models trained primarily pre-switch do not necessarily generalize post-switch; persistence can dominate | prediction_gain_y_global_honest_tail (warn on negative) | n/a | negative gain on tail evaluation |
| MNJ recovery degrades with coarser sampling | golden_a_piecewise dt_obs sweep | n/a | non-decreasing curve or weak rank correlation |
| MNPS/MNJ can separate mechanisms under matched first-order stats | golden_b2_boundary | parameter_swap | EVR/MNJ differences collapse under swap |

## golden_a_piecewise
status: **fail**

### Golden A1: Within-regime segmented prediction (primary)
Interpretation: compares data-segmented vs random split under mixed-regime eval (honest_middle).

### Golden A2: Cross-regime tail generalization (stress test)
Interpretation: tail eval can go negative; treat as WARN-level generalization stress.

metrics:
- effect_size: 0.9266132371707295
- effect_size_shuffled: -2.4336035072339035
- effect_size_random_neighbors: 0.9266132371707423
- dt_obs_sweep: [0.05]
- dt_obs_effects: [0.9999207322705576]
- dt_obs_caps: [1.0058838080080423]
- dt_obs_cap_spearman: 0.0
- dt_obs_cap_auc: 0.0
- dt_obs_cap_auc_low: 0.0
- dt_obs_cap_auc_high: 0.0
- evr: [0.6807296087622724, 0.2937942366482275, 0.02547615458950012]
- rel_mse_median: 3.8110882089369015e-05
- prediction_gain: 0.017952877951968316
- prediction_gain_shuffled: -1.208260293867307
- prediction_gain_random: 0.01795287795196554
- prediction_gain_y_local: -0.07864138904828932
- prediction_gain_y_local_delta: -0.3059311642895517
- prediction_gain_y_local_radius: -0.07864138904828932
- prediction_gain_y_local_radius_delta: -0.3059311642895517
- y_local_radius_n_median: 2.0
- y_local_radius_n_p10: 1.2
- y_local_radius_n_p90: 2.8
- y_local_window_steps: 20
- y_local_window_candidates_median: 2.0
- y_local_window_candidates_p10: 1.2
- y_local_window_candidates_p90: 2.8
- y_local_window_fallback_fraction: 1.0
- prediction_gain_y_global: 0.6835499620882295
- prediction_gain_y_global_honest: None
- prediction_gain_y_global_honest_tail: None
- prediction_gain_y_global_honest_middle: 0.868254077394871
- prediction_gain_oracle_segmented_global: 0.7064243331388786
- prediction_gain_oracle_true_segmented_global: 0.7064243331388786
- prediction_gain_oracle_true_segmented_global_honest: None
- prediction_gain_oracle_true_segmented_global_honest_tail: None
- prediction_gain_oracle_true_segmented_global_honest_middle: 0.8626079176975812
- prediction_gain_oracle_segmented_global_mse_model: 9.999265442216217e-05
- prediction_gain_oracle_segmented_global_mse_baseline: 0.0003406026636938167
- prediction_gain_oracle_segmented_split_idx: 2
- prediction_gain_oracle_segmented_split_time: 0.1
- prediction_gain_oracle_opt_segmented_global: None
- prediction_gain_oracle_opt_segmented_global_honest: None
- prediction_gain_oracle_opt_segmented_global_honest_tail: None
- prediction_gain_oracle_opt_segmented_global_honest_middle: None
- prediction_gain_oracle_opt_segmented_global_mse_model: None
- prediction_gain_oracle_opt_segmented_global_mse_baseline: None
- prediction_gain_oracle_opt_segmented_split_idx: None
- prediction_gain_oracle_opt_segmented_split_time: None
- prediction_gain_oracle_opt_segmented_sse_pre: None
- prediction_gain_oracle_opt_segmented_sse_post: None
- prediction_gain_oracle_opt_window_seconds: 0.5
- prediction_gain_oracle_opt_search_domain_start: 0
- prediction_gain_oracle_opt_search_domain_end: 3
- prediction_gain_oracle_opt_search_domain_size: 4
- prediction_gain_data_segmented_global: None
- prediction_gain_data_segmented_global_honest: None
- prediction_gain_data_segmented_global_honest_tail: None
- prediction_gain_data_segmented_global_honest_middle: None
- prediction_gain_data_segmented_global_mse_model: None
- prediction_gain_data_segmented_global_mse_baseline: None
- prediction_gain_data_segmented_global_honest_mse_model: None
- prediction_gain_data_segmented_global_honest_mse_baseline: None
- prediction_gain_data_segmented_global_honest_middle_mse_model: None
- prediction_gain_data_segmented_global_honest_middle_mse_baseline: None
- prediction_gain_data_segmented_split_idx: None
- prediction_gain_data_segmented_split_time: None
- prediction_gain_data_segmented_split_idx_honest: None
- prediction_gain_data_segmented_sse_pre: None
- prediction_gain_data_segmented_sse_post: None
- prediction_gain_data_segmented_gating_mean_pre: None
- prediction_gain_data_segmented_gating_mean_post: None
- segmentation_skipped: True
- segmentation_honest_skipped: True
- segmentation_honest_tail_skipped: True
- segmentation_honest_middle_skipped: False
- segmentation_dt_obs: 0.05
- segmentation_T: 5
- segmentation_selection_fraction: 0.7
- segmentation_fit_T: 4
- segmentation_eval_T: 1
- segmentation_eval_start: 4
- segmentation_eval_middle_start: 1
- segmentation_eval_middle_end: 4
- segmentation_fit_pairs: 3
- segmentation_data_fit_n_pre: None
- segmentation_data_fit_n_post: None
- segmentation_oracle_true_fit_n_pre: 2
- segmentation_oracle_true_fit_n_post: 1
- segmentation_oracle_opt_fit_n_pre: None
- segmentation_oracle_opt_fit_n_post: None
- segmentation_min_seg_seconds: 1.5
- segmentation_min_seg_steps: 30
- segmentation_search_domain_start: 30.0
- segmentation_search_domain_end: -26.0
- segmentation_search_domain_size: 0.0
- segmentation_eval_tail_pre_fraction: None
- segmentation_eval_tail_post_fraction: None
- segmentation_eval_middle_pre_fraction: 0.3333333333333333
- segmentation_eval_middle_post_fraction: 0.6666666666666666
- prediction_gain_segmented_shuffled: None
- prediction_gain_segmented_shuffled_honest: None
- prediction_gain_segmented_shuffled_honest_tail: None
- prediction_gain_segmented_shuffled_honest_middle: None
- prediction_gain_segmented_shuffled_mse_model: None
- prediction_gain_segmented_shuffled_mse_baseline: None
- prediction_gain_segmented_shuffled_split_idx: None
- prediction_gain_segmented_shuffled_split_time: None
- prediction_gain_segmented_shuffled_split_idx_honest: None
- prediction_gain_segmented_random_split_mean: None
- prediction_gain_segmented_random_split_std: None
- prediction_gain_segmented_random_split_draws: None
- prediction_gain_segmented_random_split_honest_mean: None
- prediction_gain_segmented_random_split_honest_std: None
- prediction_gain_segmented_random_split_honest_draws: None
- prediction_gain_segmented_random_split_honest_middle_mean: None
- prediction_gain_segmented_random_split_honest_middle_std: None
- prediction_gain_segmented_random_split_honest_middle_draws: None
- prediction_gain_segmented_dy_gate_honest_middle: None
- prediction_gain_segmented_dy_gate_honest_tail: None
- prediction_gain_segmented_dy_gate_split_idx_fit: None
- prediction_gain_segmented_dy_gate_split_time_fit: None
- dy_gate_split_objective_signal_name: dy_gate_signal
- dy_gate_split_method: gate_argmax
- dy_gate_signal_p10: 0.017098515948215508
- dy_gate_signal_p50: 0.023316021636377318
- dy_gate_signal_p90: 0.03552926509838968
- dy_gate_signal_nonzero_fraction: 1.0
- dy_gate_signal_mean: 0.02545735655560966
- dy_gate_signal_std: 0.009552045271465227
- oracle_true_split_idx_fit: 2
- dy_gate_gate_argmax_idx_fit: None
- dy_gate_gate_argmax_time_fit: None
- dy_gate_gate_argmax_value: None
- dy_gate_gate_argmax_abs_err_vs_oracle: None
- dy_gate_split_sse_total: None
- dy_gate_split_sse_pre: None
- dy_gate_split_sse_post: None
- prediction_gain_segmented_mnj_gate_honest_middle: None
- prediction_gain_segmented_mnj_gate_honest_tail: None
- prediction_gain_segmented_mnj_gate_split_idx_fit: None
- prediction_gain_segmented_mnj_gate_split_time_fit: None
- mnj_gate_split_objective_signal_name: mnj_gate_signal
- mnj_gate_split_method: gate_argmax
- mnj_gate_split_sse_total: None
- mnj_gate_split_sse_pre: None
- mnj_gate_split_sse_post: None
- prediction_gain_segmented_mnj_gate_shuffled_honest_middle: None
- prediction_gain_segmented_mnj_gate_shuffled_honest_tail: None
- mnj_gate_trust_coverage: 0.0
- mnj_gate_trust_score_p10: 0.5
- mnj_gate_trust_score_p50: 0.5
- mnj_gate_trust_score_p90: 0.5
- mnj_gate_trust_score_nonzero_fraction: 1.0
- mnj_gate_residual_weight_p10: 0.999999044901545
- mnj_gate_residual_weight_p50: 0.99999952068854
- mnj_gate_residual_weight_p90: 0.9999999648029082
- mnj_gate_residual_weight_nonzero_fraction: 1.0
- mnj_gate_pass_residual_fraction: 1.0
- mnj_gate_rel_mse_threshold: 0.5
- mnj_gate_signal_p10: 0.5324389199258882
- mnj_gate_signal_p50: 0.943490804617796
- mnj_gate_signal_p90: 1.3415142492910868
- mnj_gate_signal_nonzero_fraction: 1.0
- mnj_gate_signal_mean: 0.9388377903254327
- mnj_gate_signal_std: 0.39400948192859336
- mnj_gate_gate_argmax_idx_fit: None
- mnj_gate_gate_argmax_time_fit: None
- mnj_gate_gate_argmax_value: None
- mnj_gate_gate_argmax_abs_err_vs_oracle: None
- mnj_gate_rel_mse_p10: 3.519709333104059e-08
- mnj_gate_rel_mse_p50: 4.793118661988341e-07
- mnj_gate_rel_mse_p90: 9.550993685479088e-07
- mnj_gate_cond_p10: 25027.46170682172
- mnj_gate_cond_p50: 25454.389363633818
- mnj_gate_cond_p90: 35202.84235944849
- mnj_gate_neighbors_p10: 3.0
- mnj_gate_neighbors_p50: 3.0
- mnj_gate_neighbors_p90: 3.0
- mnj_gate_rank_p10: 1.000000000001856
- mnj_gate_rank_p50: 1.3772532099966925
- mnj_gate_rank_p90: 1.7840592181032922
- mnj_gate_excitation_p10: 0.7900432934152013
- mnj_gate_excitation_p50: 0.9340693716728324
- mnj_gate_excitation_p90: 0.9515900981354222
- mnj_gate_fail_rank_fraction: 0.0
- mnj_gate_fail_excitation_fraction: 0.0
- mnj_gate_fail_residual_fraction: 0.0
- mnj_gate_fail_condition_fraction: 1.0
- mnj_gate_fail_neighbors_fraction: 1.0
- mnj_gate_derivative_method: discrete_step
- mnj_gate_margin: 0.01
- oracle_opt_margin: 0.01
- prediction_gain_oracle_ceiling: 0.9940718046259304
- prediction_gain_model_vs_oracle_gap: 0.00033246870748321973
- prediction_gain_mse_model: 0.0003344878666244534
- prediction_gain_mse_baseline: 0.0003406026636938167
- prediction_gain_mse_oracle: 2.0191591412336744e-06
- grid_pass_rate: 0.0
- grid_leak_rate: 1.0

dt_obs_curve:
| dt_obs | score |
| --- | --- |
| 0.05 | 0.999921 |

honest_middle:
| model | gain | mse_model | mse_baseline |
| --- | --- | --- | --- |
| global | 0.868254077394871 | None | None |
| oracle_true | 0.8626079176975812 | None | None |
| oracle_opt | None | None | None |
| data_segmented | None | None | None |
| segmented_shuffled | None | None | None |
| random_split_mean | None | None | None |
| dy_gate | None | None | None |
| mnj_gate | None | None | None |
| mnj_gate_shuffled | None | None | None |

honest_middle_split: selection_fraction=0.7, fit_T=4, eval_start=1, eval_end=4

honest_tail:
| model | gain | mse_model | mse_baseline |
| --- | --- | --- | --- |
| global | None | None | None |
| oracle_true | None | None | None |
| oracle_opt | None | None | None |
| data_segmented | None | None | None |
| segmented_shuffled | None | None | None |
| random_split_mean | None | None | None |
| dy_gate | None | None | None |
| mnj_gate | None | None | None |
| mnj_gate_shuffled | None | None | None |

honest_tail_split: selection_fraction=0.7, fit_T=4, eval_start=4, eval_T=1

checks:
- effect_size_min: True
- time_shuffle_max: False
- random_neighbors_max: False
- grid_pass_rate_min: False
- grid_leak_rate_max: False
- segmented_shuffled_collapse: True
- segmented_random_split_beaten: True
- mnj_gate_beats_dy_gate: True
- mnj_gate_beats_random: True
- mnj_gate_shuffled_collapse: True
- oracle_opt_ge_oracle_true: True
- data_honest_le_oracle_opt: True
- global_honest_tail_nonnegative: True
