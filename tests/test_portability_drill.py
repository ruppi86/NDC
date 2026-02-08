import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def test_portability_drill_runs_with_analysis_only(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    analysis_root = tmp_path / "analysis_root"
    analysis_pkg = analysis_root / "ndc_analysis"
    shutil.copytree(repo_root / "ndc_analysis", analysis_pkg)

    obs_path = tmp_path / "obs.h5"
    with h5py.File(obs_path, "w") as h5:
        t = np.linspace(0.0, 0.2, 11)
        Y = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
        h5.create_dataset("t", data=t)
        h5.create_dataset("Y", data=Y)
        h5.attrs["metadata"] = json.dumps({"file_format_version": "observation_export_v1"})

    script = repo_root / "scripts" / "run_portability_drill.py"
    out_path = tmp_path / "out.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(analysis_root)

    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--analysis-root",
            str(analysis_root),
            "--observations",
            str(obs_path),
            "--out",
            str(out_path),
        ],
        cwd=tmp_path,
        env=env,
    )

    assert out_path.exists()
