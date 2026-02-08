from pathlib import Path

from NDC.config.schema import load_config


def test_load_baseline_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "NDC" / "config" / "presets" / "baseline.yaml")
    assert cfg.latent_dim == 10
