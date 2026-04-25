from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "src"))

import argparse
import logging

from airpollution.logging_utils import setup_logging
from airpollution.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PM baseline model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()
    metrics = run_training_pipeline(args.config)
    print(metrics)


if __name__ == "__main__":
    main()
