import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_manifest_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_benchmarks.py"
    manifest = repo_root / "benchmarks" / "manifest.yaml"
    out_dir = tmp_path / "benchmarks"
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--smoke",
            "--max-benchmarks",
            "1",
        ],
        cwd=repo_root,
    )
    assert (out_dir / "benchmark_report.md").exists()
    json_path = out_dir / "benchmark_report.json"
    assert json_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report.get("report_version") == "0.1"
    assert report.get("manifest_version") == "0.1"
    results = report.get("results") or {}
    assert results
    golden = results.get("golden_a_piecewise") or next(iter(results.values()))
    metrics = golden.get("metrics") or {}
    checks = golden.get("checks") or {}
    for key in [
        "mnj_gate_split_method",
        "dy_gate_split_method",
        "mnj_gate_gate_argmax_idx_fit",
        "oracle_true_split_idx_fit",
        "mnj_gate_margin",
    ]:
        assert key in metrics
    assert "mnj_gate_beats_random" in checks
