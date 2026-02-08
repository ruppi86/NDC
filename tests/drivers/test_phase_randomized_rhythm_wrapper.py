import json
from pathlib import Path

import numpy as np

from NDC.drivers.file_timeseries import FileTimeseriesDriver
from NDC.drivers.surrogates import PhaseRandomizedRhythmWrapperDriver


def test_phase_randomized_rhythm_preserves_control(tmp_path: Path):
    p = tmp_path / "trace.json"
    data = {
        "samples": [
            {"t": 0.0, "rhythm": 0.0, "control": [0.0, 0.0, 0.0]},
            {"t": 1.0, "rhythm": 1.0, "control": [1.0, 2.0, 3.0]},
            {"t": 2.0, "rhythm": 0.0, "control": [2.0, 4.0, 6.0]},
        ]
    }
    p.write_text(json.dumps(data), encoding="utf-8")

    base = FileTimeseriesDriver(path=str(p), latent_dim=3, file_format="json")
    rng = np.random.default_rng(123)
    wrapped = PhaseRandomizedRhythmWrapperDriver.from_driver(
        base_driver=base, t_start=0.0, t_end=2.0, dt=0.5, rng=rng
    )

    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        base_state = base(t)
        wrapped_state = wrapped(t)
        # rhythm should generally differ (not asserted deterministically), but control must match exactly.
        if base_state.control is None:
            assert wrapped_state.control is None
        else:
            np.testing.assert_allclose(wrapped_state.control, base_state.control)

