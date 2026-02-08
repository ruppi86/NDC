import json
from pathlib import Path

import h5py

from NDC.config.schema import (
    DriverConfig,
    ExperimentConfig,
    LandscapeConfig,
    ObserverConfig,
    RVCConfig,
    TimeConfig,
)
from NDC.experiments.runner import run_experiment_from_config


def test_observation_export_contract(tmp_path: Path):
    cfg = ExperimentConfig(
        seed=1,
        latent_dim=3,
        initial_state=[0.0, 0.0, 0.0],
        time=TimeConfig(t_start=0.0, t_end=0.2, dt_sim=0.01, dt_obs=0.02, window=0.1, step=0.05),
        landscape=LandscapeConfig(name="quadratic_well", params={}),
        rvc=RVCConfig(name="null_gate", params={"sigma0": 0.0}),
        driver=DriverConfig(name="sine_rhythm", params={"amplitude": 0.0, "frequency": 1.0}),
        observer=ObserverConfig(name="oracle", params={}),
    )
    cfg.output.output_dir = str(tmp_path)
    cfg.output.file_prefix = "boundary"

    run_experiment_from_config(cfg, output_dir=tmp_path)

    obs_path = tmp_path / "boundary_observations.h5"
    assert obs_path.exists()

    with h5py.File(obs_path, "r") as h5:
        keys = set(h5.keys())
        allowed = {"t", "Y", "rhythm", "control", "win_t", "win_Y"}
        assert keys.issubset(allowed)
        assert "t" in keys and "Y" in keys
        assert "states" not in keys

        meta_raw = h5.attrs.get("metadata", "{}")
        meta = json.loads(meta_raw)
        for field in [
            "dt_sim",
            "dt_obs",
            "window",
            "step",
            "downsample_method",
            "observer_tier",
            "observer_params",
            "seed",
            "config_hash",
            "file_format_version",
        ]:
            assert field in meta
