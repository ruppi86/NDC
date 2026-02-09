from pathlib import Path

import yaml

from NDC.experiments.runner import run_experiment


def test_trivial_run(tmp_path: Path):
    """Smoke test to ensure the simulator can run with a minimal configuration."""
    cfg = {
        "seed": 1,
        "latent_dim": 3,
        "initial_state": [0, 0, 0],
        "time": {
            "t_start": 0.0,
            "t_end": 0.05,
            "dt_sim": 0.01,
            "dt_obs": 0.02,
            "window": 0.04,
            "step": 0.02,
        },
        "landscape": {"name": "quadratic_well", "params": {}},
        "rvc": {"name": "null_gate", "params": {"sigma0": 0.0}},
        "driver": {"name": "sine_rhythm", "params": {"amplitude": 0.0, "frequency": 1.0}},
        "observer": {"name": "oracle", "params": {}},
        "plasticity": {"name": "static", "params": {}},
        "scheduler": {"default_label": "wake", "regimes": []},
        "output": {"output_dir": str(tmp_path), "file_prefix": "trivial"},
        "notes": {},
    }
    config_path = tmp_path / "trivial.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    results = run_experiment(config_path)
    assert "oracle_metrics" in results
