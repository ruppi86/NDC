import numpy as np

from NDC.dynamics.state import Trajectory
from NDC.io.trajectory import load_trajectory, save_trajectory


def test_trajectory_roundtrip(tmp_path):
    times = np.array([0.0, 0.1, 0.2])
    states = np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]])
    rhythm = np.array([0.0, 0.5, 1.0])

    traj = Trajectory(times=times, states=states, rhythm=rhythm)
    path = tmp_path / "traj.h5"
    save_trajectory(path, traj, metadata={"test": True})

    loaded, meta = load_trajectory(path)
    assert np.allclose(loaded.times, times)
    assert np.allclose(loaded.states, states)
    assert np.allclose(loaded.rhythm, rhythm)
    assert meta.get("test") is True
