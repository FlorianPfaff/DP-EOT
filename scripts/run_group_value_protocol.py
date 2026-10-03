"""Run the frozen protocol in order; each stage resumes from verified checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")})
    calibration = args.output_root / "calibrate/calibration.json"
    stages = (
        ("pilot", []),
        ("calibrate", []),
        ("confirm", ["--calibration", str(calibration)]),
        ("beam16", ["--calibration", str(calibration), "--num-trials", "20", "--hypotheses", "16"]),
        ("runtime", ["--calibration", str(calibration)]),
    )
    for name, extra in stages:
        stage = "confirm" if name == "beam16" else name
        workers = 1 if name == "runtime" else args.workers
        print(f"Starting protocol stage: {name}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "dpeot.experiments.export_group_value_study",
             "--stage", stage, "--workers", str(workers), "--output-dir", str(args.output_root / name), *extra],
            env=environment, check=True,
        )
    print("All prescribed stages completed. Audit artifacts before drawing conclusions.", flush=True)


if __name__ == "__main__":
    main()
