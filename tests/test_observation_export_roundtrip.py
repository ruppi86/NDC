import numpy as np

from NDC.io.observation_export import (
    ObservationExportMetadata,
    export_observations,
    load_observations,
)


def test_observation_export_roundtrip(tmp_path):
    times = np.array([0.0, 0.1, 0.2])
    values = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    win_t = np.array([0.1])
    win_y = np.array([[1.1, 2.1]])
    meta = ObservationExportMetadata(
        dt_sim=0.01,
        dt_obs=0.1,
        window=0.2,
        step=0.1,
        downsample_method="interp",
        observer_tier="obs-0",
        observer_params={},
        seed=1,
        config_hash="abc123",
    )

    path = tmp_path / "obs.h5"
    export_observations(
        path,
        times=times,
        values=values,
        metadata=meta,
        windowed_times=win_t,
        windowed_values=win_y,
    )

    data, loaded_meta = load_observations(path)
    assert np.allclose(data["t"], times)
    assert np.allclose(data["Y"], values)
    assert np.allclose(data["win_t"], win_t)
    assert np.allclose(data["win_Y"], win_y)
    assert loaded_meta["config_hash"] == "abc123"
