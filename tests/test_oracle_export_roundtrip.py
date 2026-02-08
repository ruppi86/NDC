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


def test_oracle_export_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig(
        seed=2,
        latent_dim=3,
        initial_state=[0.0, 0.0, 0.0],
        time=TimeConfig(t_start=0.0, t_end=0.2, dt_sim=0.01, dt_obs=0.02, window=0.1, step=0.05),
        landscape=LandscapeConfig(name="quadratic_well", params={}),
        rvc=RVCConfig(name="null_gate", params={"sigma0": 0.0}),
        driver=DriverConfig(name="sine_rhythm", params={"amplitude": 0.0, "frequency": 1.0}),
        observer=ObserverConfig(name="oracle", params={}),
    )
    cfg.output.output_dir = str(tmp_path)
    cfg.output.file_prefix = "oracle"
    cfg.output.save_oracle = True

    run_experiment_from_config(cfg, output_dir=tmp_path)

    oracle_path = tmp_path / "oracle_oracle.h5"
    assert oracle_path.exists()

    with h5py.File(oracle_path, "r") as h5:
        assert "t" in h5 and "X" in h5
        assert h5["t"].ndim == 1
        assert h5["X"].ndim == 2
        assert "A_true" in h5
        assert h5["A_true"].ndim == 3
        meta_raw = h5.attrs.get("metadata", "{}")
        assert "oracle_export_v1" in meta_raw
