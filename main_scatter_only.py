# -*- coding: utf-8 -*-
"""
Scatter-only pipeline runner (without HPA).

Runs:
1) Root scatter script: Fullgraph_MeAc.py
2) DIAM-separated scatter script: diam_separated/Fullgraph_MeAc.py for each DIAM

Usage:
    python main_scatter_only.py
    python main_scatter_only.py --skip-main
    python main_scatter_only.py --skip-diam
    python main_scatter_only.py --diam-values 1 2 3
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "new_data.xlsx"
DEFAULT_DIAM_VALUES = [1, 1.5, 2, 2.5, 3]


def run_script(label, script, env):
    script_path = ROOT / script
    if not script_path.exists():
        print(f"\n[{label}] FAILED - script not found: {script_path}")
        return False, 0.0

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "done" if success else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status} - {elapsed:.1f}s")
    return success, elapsed


def make_base_env():
    env = dict(os.environ)
    env["KARISA_USE_HPA"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("KARISA_DIAM_FILTER", None)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Run only scatter plot scripts.")
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Skip root scatter script (Fullgraph_MeAc.py).",
    )
    parser.add_argument(
        "--skip-diam",
        action="store_true",
        help="Skip DIAM-separated scatter script.",
    )
    parser.add_argument(
        "--diam-values",
        type=float,
        nargs="+",
        default=DEFAULT_DIAM_VALUES,
        help=f"DIAM values for DIAM-separated scatter runs (default: {DEFAULT_DIAM_VALUES}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    if args.skip_main and args.skip_diam:
        print("Nothing to run: both --skip-main and --skip-diam were provided.")
        sys.exit(1)

    print("=" * 80)
    print("  KARISA PROJECT - SCATTER-ONLY PIPELINE (WITHOUT HPA)")
    print(f"  Dataset: {DATASET_PATH.relative_to(ROOT)}")
    if not args.skip_diam:
        print(f"  DIAM values: {args.diam_values}")
    print("=" * 80)

    total_start = time.time()
    results = []

    if not args.skip_main:
        env_main = make_base_env()
        success, elapsed = run_script(
            label="Main Scatter (Fullgraph_MeAc.py)",
            script="Fullgraph_MeAc.py",
            env=env_main,
        )
        results.append(("Main Scatter", success, elapsed))

    if not args.skip_diam:
        for diam_val in args.diam_values:
            env_diam = make_base_env()
            env_diam["KARISA_DIAM_FILTER"] = str(diam_val)
            diam_label = f"DIAM_{diam_val:g}"
            success, elapsed = run_script(
                label=f"DIAM-Separated Scatter ({diam_label})",
                script="diam_separated/Fullgraph_MeAc.py",
                env=env_diam,
            )
            results.append((f"Scatter {diam_label}", success, elapsed))

    total = time.time() - total_start

    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for label, success, elapsed in results:
        mark = "OK" if success else "FAIL"
        print(f"  [{mark}]  {label:<35}  {elapsed:6.1f}s")
    print(f"{'=' * 80}")
    print(f"  Total: {total:.1f}s")
    print(f"{'=' * 80}\n")

    all_ok = all(success for _, success, _ in results) if results else False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
