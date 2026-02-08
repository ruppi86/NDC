from pathlib import Path
import sys


def test_resolution_stress_rows(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.run_ndc_resolution_stress_grid import run_resolution_stress

    rows = run_resolution_stress(
        base_preset=root / "NDC" / "config" / "presets" / "file_input_example.yaml",
        out_root=tmp_path / "outputs",
        input_dir=tmp_path / "inputs",
        latent_dim=3,
        t_start=0.0,
        t_end=0.1,
        dt_input=0.01,
        dt_sim=0.01,
        dt_obs_values=[0.05],
        freqs=[0.2],
        step_times=[0.05],
        step_mags=[0.1],
        seed=1,
        n_runs=1,
        window_min=0.05,
        window_mult=2,
        event_window=0.05,
    )

    assert len(rows) == 1
    row = rows[0]
    assert "dt_obs" in row and "rhythm_freq" in row
    assert "delta_speed_mean" in row and "es_speed_mean" in row
    assert "event_delta_speed_mean" in row and "event_es_speed_mean" in row