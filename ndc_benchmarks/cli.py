"""CLI entrypoint for benchmark suite."""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
import json
from pathlib import Path

from .golden_a import _golden_a
from .golden_b2 import _golden_b2
from .invariance import _invariance_stress
from .io import _load_claims, _load_manifest
from .report import _build_sections, _render_report
from .types import BenchmarkResult, REPORT_VERSION


def main() -> None:
    """Main entrypoint for the NDC benchmark suite CLI.

    Loads the manifest and claims, runs each specified benchmark,
    and renders a report to the output directory.
    """
    p = argparse.ArgumentParser(description="Run NDC benchmark suite.")
    p.add_argument("--manifest", default="benchmarks/manifest.yaml")
    p.add_argument("--out-dir", default="outputs/benchmarks")
    p.add_argument("--smoke", action="store_true", help="Use small timebase and fewer sweeps.")
    p.add_argument("--max-benchmarks", type=int, default=None)
    p.add_argument("--fail-on", action="store_true", help="Exit non-zero on any failure.")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes to use for per-(dt_obs,seed) loops. Default 1.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print per-step progress for long-running benchmarks (dt_obs/seed loops).",
    )
    args = p.parse_args()

    def log(msg: str) -> None:
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        print(f"[{ts}] {msg}", flush=True)

    if args.progress:
        os.environ["NDC_BENCH_PROGRESS"] = "1"
        log("Progress logging enabled (NDC_BENCH_PROGRESS=1).")
    # Propagate jobs to benchmark runners (used where supported).
    os.environ["NDC_BENCH_JOBS"] = str(max(1, int(args.jobs)))
    log(f"Parallel jobs: {os.environ['NDC_BENCH_JOBS']}")

    t0 = time.perf_counter()
    log(f"Loading manifest: {args.manifest}")
    manifest = _load_manifest(Path(args.manifest))
    log("Loading claims: benchmarks/claims.yaml")
    claims = _load_claims(Path("benchmarks/claims.yaml"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output dir: {out_dir.resolve()}")

    results: dict[str, BenchmarkResult] = {}
    benchmarks = manifest["benchmarks"]
    if args.max_benchmarks is not None:
        benchmarks = benchmarks[: int(args.max_benchmarks)]

    log(
        f"Running {len(benchmarks)} benchmark(s) "
        f"(smoke={bool(args.smoke)}, max_benchmarks={args.max_benchmarks})."
    )
    for bench in benchmarks:
        btype = bench.get("type")
        bid = bench.get("id", btype)
        b0 = time.perf_counter()
        log(f"START benchmark id={bid} type={btype}")
        if btype == "golden_a":
            results[bid] = _golden_a(
                bench,
                out_dir=out_dir,
                smoke=args.smoke,
                jobs=max(1, int(args.jobs)),
            )
        elif btype == "golden_b2":
            results[bid] = _golden_b2(bench, out_dir=out_dir, smoke=args.smoke)
        elif btype == "invariance_stress":
            results[bid] = _invariance_stress(
                bench,
                out_dir=out_dir,
                smoke=args.smoke,
                jobs=max(1, int(args.jobs)),
            )
        else:
            raise ValueError(f"Unknown benchmark type: {btype}")
        b1 = time.perf_counter()
        log(f"END benchmark id={bid} status={results[bid].status} elapsed_s={b1 - b0:.2f}")

    log("Rendering Markdown report.")
    report = _render_report(results, claims)
    report_path = out_dir / "benchmark_report.md"
    json_path = out_dir / "benchmark_report.json"
    log(f"Writing: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    log("Building JSON sections.")
    sections = _build_sections(results)
    log(f"Writing: {json_path}")
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
    log("Writing copies to reports/ (benchmark_v1.md / benchmark_v1.json).")
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

    t1 = time.perf_counter()
    log(f"Done. Total elapsed_s={t1 - t0:.2f}")

    if args.fail_on and any(r.status == "fail" for r in results.values()):
        raise SystemExit(1)
