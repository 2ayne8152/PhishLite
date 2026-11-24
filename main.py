"""
main.py - unified CLI entry point for the PhishLite pipeline (Part A + Part B).

Examples:
    python main.py part_a                       # eda -> baseline
    python main.py sentinel                     # run step 1 of Part B, then check SENTINEL_CHOICE
    python main.py part_b                       # sentinel -> scaling -> collinearity -> reduction -> tuning -> ensemble -> eval
    python main.py tuning --models svm mlp      # tune only selected models
    python main.py ensemble                     # run step 10 only
    python main.py eval                         # run steps 11-13 only
    python main.py all --data path/to/data.csv  # full pipeline
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import load
from a_baseline import run_baseline
from a_eda import run_eda
from b_ensemble import run_ensemble
from b_evaluation import run_eval
from b_experiments import (
    run_collinearity,
    run_reduction,
    run_scaling,
    run_sentinel,
)
import b_tuning

PART_A_STAGES = ["eda", "baseline"]
PART_B_STAGES = ["sentinel", "scaling", "collinearity", "reduction", "tuning", "ensemble", "eval"]
LOG_FILE = Path("reports/run.log")


class _Tee:
    """Write output to both the terminal and reports/run.log (skipping spinner carriage returns)."""

    def __init__(self, stream, log_handle):
        self._stream = stream
        self._log = log_handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        if not text.startswith("\r"):
            self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def main() -> None:
    all_stages = PART_A_STAGES + PART_B_STAGES
    choices = [*all_stages, "part_a", "part_b", "all"]

    parser = argparse.ArgumentParser(description="PhishLite Part A & Part B automated pipeline")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=choices,
        help="which stage or group to run (default: all)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help=f"path to dataset_full.csv (default: {load.FULL_FILE})",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="optional model slugs for the tuning stage (e.g. --models svm mlp)",
    )
    args = parser.parse_args()

    if args.data:
        load.FULL_FILE = args.data

    stages = {
        "eda": run_eda,
        "baseline": run_baseline,
        "sentinel": run_sentinel,
        "scaling": run_scaling,
        "collinearity": run_collinearity,
        "reduction": run_reduction,
        "tuning": lambda: b_tuning.main(args.models),
        "ensemble": run_ensemble,
        "eval": run_eval,
    }

    if args.stage == "all":
        todo = all_stages
    elif args.stage == "part_a":
        todo = PART_A_STAGES
    elif args.stage == "part_b":
        todo = PART_B_STAGES
    else:
        todo = [args.stage]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    with LOG_FILE.open("a", encoding="utf-8") as log_f:
        sys.stdout = _Tee(orig_stdout, log_f)
        sys.stderr = _Tee(orig_stderr, log_f)
        try:
            print(f"\n{'#' * 70}")
            print(f"RUN START : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"STAGE(S)  : {', '.join(todo)}")
            print(f"SEED      : {load.make_seeds(n=3, master=load.RANDOM_SEED)}")
            print(f"DATASET   : {Path(load.FULL_FILE).resolve()}")
            print(f"{'#' * 70}")

            total_start = time.time()
            for stage in todo:
                print(f"\n{'=' * 70}\nSTAGE: {stage}\n{'=' * 70}")
                stage_start = time.time()
                try:
                    stages[stage]()
                    print(f"\n[{stage} DONE] elapsed: {time.time() - stage_start:.1f}s")
                except FileNotFoundError as exc:
                    print(f"[FILE ERROR] {exc}")
                    if stage == "eda" or args.stage not in {"all", "part_a", "part_b"}:
                        break
                except SystemExit as exc:
                    print(f"[{stage} STOPPED] {exc}")
                    break
                except Exception as exc:
                    print(f"[{stage} FAILED] {type(exc).__name__}: {exc}")
                    traceback.print_exc()

            print(f"\nPipeline finished in {time.time() - total_start:.1f}s. Log saved -> {LOG_FILE.resolve()}")
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


if __name__ == "__main__":
    main()