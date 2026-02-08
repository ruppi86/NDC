"""CLI entrypoint for benchmark suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .golden_a import _golden_a
from .golden_b2 import _golden_b2
from .invariance import _invariance_stress
from .io import _load_claims, _load_manifest
from .report import _build_sections, _render_report
from .types import BenchmarkResult, REPORT_VERSION


def main() -> None:
    p = argparse.ArgumentParser(description="Run NDC benchmark suite.")
    p.add_argument("--manifest", default="benchmarks/manifest.yaml")
    p.add_argument("--out-dir", default="outputs/benchmarks")
    p.add_argument("--smoke", action="store_true", help="Use small timebase and fewer sweeps.")
    p.add_argument("--max-benchmarks", type=int, default=None)
    p.add_argument("--fail-on", action="store_true", help="Exit non-zero on any failure.")
    args = p.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    claims = _load_claims(Path("benchmarks/claims.yaml"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, BenchmarkResult] = {}
    benchmarks = manifest["benchmarks"]
    if args.max_benchmarks is not None:
        benchmarks = benchmarks[: int(args.max_benchmarks)]

    for bench in benchmarks:
        btype = bench.get("type")
        bid = bench.get("id", btype)
        if btype == "golden_a":
            results[bid] = _golden_a(bench, out_dir=out_dir, smoke=args.smoke)
        elif btype == "golden_b2":
            results[bid] = _golden_b2(bench, out_dir=out_dir, smoke=args.smoke)
        elif btype == "invariance_stress":
            results[bid] = _invariance_stress(bench, out_dir=out_dir, smoke=args.smoke)
        else:
            raise ValueError(f"Unknown benchmark type: {btype}")

    report = _render_report(results, claims)
    report_path = out_dir / "benchmark_report.md"
    json_path = out_dir / "benchmark_report.json"
    report_path.write_text(report, encoding="utf-8")
    sections = _build_sections(results)
    json_path.write_text(
        json.dumps(
            {
                "report_version": REPORT_VERSION,
                "manifest_version": manifest.get("manifest_version"),
                "claims": claims,
                "sections": sections,
                "results": {k: result.__dict__ for k, result in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.joinpath("benchmark_v1.md").write_text(report, encoding="utf-8")
    reports_dir.joinpath("benchmark_v1.json").write_text(
        json.dumps(
            {
                "report_version": REPORT_VERSION,
                "manifest_version": manifest.get("manifest_version"),
                "claims": claims,
                "sections": sections,
                "results": {k: result.__dict__ for k, result in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.fail_on and any(r.status == "fail" for r in results.values()):
        raise SystemExit(1)
