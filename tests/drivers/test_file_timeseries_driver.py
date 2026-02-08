import json
from pathlib import Path

import numpy as np

from NDC.drivers.file_timeseries import FileTimeseriesDriver


def test_file_timeseries_driver_json_interpolation(tmp_path: Path):
    p = tmp_path / "trace.json"
    data = {
        "samples": [
            {"t": 0.0, "rhythm": 0.0, "control": [0.0, 0.0, 0.0]},
            {"t": 1.0, "rhythm": 1.0, "control": [1.0, 2.0, 3.0]},
        ]
    }
    p.write_text(json.dumps(data), encoding="utf-8")

    drv = FileTimeseriesDriver(path=str(p), latent_dim=3, file_format="json")
    state = drv(0.5)
    assert np.isclose(state.rhythm, 0.5)
    assert state.control is not None
    np.testing.assert_allclose(state.control, np.array([0.5, 1.0, 1.5]))


def test_file_timeseries_driver_csv_no_control(tmp_path: Path):
    p = tmp_path / "trace.csv"
    p.write_text("t,rhythm\n0.0,0.0\n1.0,1.0\n", encoding="utf-8")

    drv = FileTimeseriesDriver(path=str(p), latent_dim=10, file_format="csv")
    state = drv(0.25)
    assert np.isclose(state.rhythm, 0.25)
    assert state.control is None

