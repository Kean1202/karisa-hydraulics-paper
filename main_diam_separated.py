# -*- coding: utf-8 -*-
"""
DIAM-Separated Pipeline

Runs the full analysis pipeline (without HPA) once per valid DIAM value.
Each run filters the dataset to a single DIAM, uses 6 independent variables
(DIAM excluded since it's constant), and writes outputs to:

    results/diam_separated/DIAM_{value}/

Run:
    python main_diam_separated.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "new_data.xlsx"

# Valid DIAM values to iterate over
DIAM_VALUES = [1, 1.5, 2, 2.5, 3]

STEPS = [
    ("Variable Importance (RF + XGBoost)", "diam_separated/best_models_analysis.py"),
    ("Interaction Diagrams",               "diam_separated/interaction_diagrams.py"),
    ("Discrete Contour Maps",              "diam_separated/discrete_contour_map.py"),
    ("MeAc Full Scatter Graph",            "diam_separated/Fullgraph_MeAc.py"),
]


def run_step(idx, label, script, env, n_steps):
    script_path = ROOT / script
    if not script_path.exists():
        print(f"\n[{label}] FAILED - script not found: {script_path}")
        return False, 0.0

    print(f"\n{'=' * 80}")
    print(f"  STEP {idx + 1}/{n_steps}: {label}")
    print(f"{'=' * 80}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "done" if success else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status} - {elapsed:.1f}s")
    return success, elapsed


def run_for_diam(diam_val):
    diam_label = f"DIAM_{diam_val:g}"

    print(f"\n{'#' * 80}")
    print(f"#  STARTING RUN: {diam_label}")
    print(f"{'#' * 80}")

    env = dict(os.environ)
    env["KARISA_USE_HPA"] = "0"
    env["KARISA_DIAM_FILTER"] = str(diam_val)
    env["PYTHONIOENCODING"] = "utf-8"

    total_start = time.time()
    results = []

    for idx, (label, script) in enumerate(STEPS):
        success, elapsed = run_step(idx, label, script, env, len(STEPS))
        results.append((label, success, elapsed))
        if not success:
            print(f"\nPipeline for {diam_label} stopped at step {idx + 1} due to failure.")
            break

    total = time.time() - total_start

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY  [{diam_label}]")
    print(f"{'=' * 80}")
    for label, success, elapsed in results:
        mark = "OK" if success else "FAIL"
        print(f"  [{mark}]  {label:<45}  {elapsed:6.1f}s")
    print(f"{'=' * 80}")
    print(f"  Total for {diam_label}: {total:.1f}s")
    print(f"{'=' * 80}\n")

    return all(s for _, s, _ in results)


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    print("=" * 80)
    print("  KARISA PROJECT - DIAM-SEPARATED PIPELINE (WITHOUT HPA)")
    print(f"  Dataset: {DATASET_PATH.relative_to(ROOT)}")
    print(f"  DIAM values: {DIAM_VALUES}")
    print(f"  Independent variables per run: 6  (NHOLES, HDIAM, TRAYSPC, WEIRHT, DECK, NPASS)")
    print("=" * 80)

    overall_start = time.time()
    diam_results = {}

    for diam_val in DIAM_VALUES:
        ok = run_for_diam(diam_val)
        diam_results[f"DIAM_{diam_val:g}"] = ok

    overall_total = time.time() - overall_start

    print(f"\n{'#' * 80}")
    print("  OVERALL SUMMARY - DIAM-SEPARATED PIPELINE")
    print(f"{'#' * 80}")
    for diam_label, ok in diam_results.items():
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}]  {diam_label}")
    print(f"{'#' * 80}")
    print(f"  Grand total: {overall_total:.1f}s")
    print(f"{'#' * 80}\n")

    all_ok = all(diam_results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
