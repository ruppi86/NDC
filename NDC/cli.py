"""Command-line entrypoint for synthetic NDC runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from NDC.experiments.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the NDC CLI.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run a synthetic NDC experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser


def main() -> None:
    """Main entrypoint for the NDC CLI.

    Parses command-line arguments and runs the specified experiment.
    """
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(Path(args.config), output_dir=args.output_dir)


if __name__ == "__main__":
    main()

