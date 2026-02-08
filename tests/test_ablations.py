from pathlib import Path

from NDC.config.schema import DriverConfig, ExperimentConfig, LandscapeConfig, RVCConfig, TimeConfig
from NDC.experiments.ablations import run_ablation_comparison


def test_gate_ablation_detects_difference(tmp_path: Path):
    cfg = ExperimentConfig(
        seed=42,
        latent_dim=3,
        initial_state=[0.0, 0.0, 0.0],
        time=TimeConfig(t_start=0.0, t_end=0.3, dt_sim=0.01, dt_obs=0.02, window=0.1, step=0.05),
        landscape=LandscapeConfig(name="quadratic_well", params={}),
        rvc=RVCConfig(name="linear_gate", params={"sigma_min": 0.0, "sigma_max": 1.0}),
        driver=DriverConfig(name="sine_rhythm", params={"amplitude": 1.0, "frequency": 2.0, "offset": 0.5}),
    )

    counter = cfg.model_copy(deep=True)
    counter.rvc = RVCConfig(name="null_gate", params={"sigma0": 0.0})

    result = run_ablation_comparison(cfg, counter, tmp_path, n_runs=2)
    assert "speed_mean" in result.deltas
    assert abs(result.deltas["speed_mean"]) > 1e-6 or abs(result.deltas["state_var"]) > 1e-6
