# -*- coding: utf-8 -*-
"""
Main pipeline (WITH HPA).

Run:
    python main_with_hpa.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "new_data.xlsx"

STEPS = [
    ("Variable Importance (RF + XGBoost)", "best_models_analysis.py"),
    ("Interaction Diagrams", "interaction_diagrams.py"),
    ("Discrete Contour Maps", "discrete_contour_map.py"),
    ("MeAc Full Scatter Graph", "Fullgraph_MeAc.py"),
    ("HPA Sweet-Spot Visualization", "hpa_sweet_spot.py"),
]


def run_step(idx, label, script, env):
    script_path = ROOT / script
    if not script_path.exists():
        print(f"\n[{label}] FAILED - script not found: {script_path}")
        return False, 0.0

    print(f"\n{'=' * 80}")
    print(f"  STEP {idx + 1}/{len(STEPS)}: {label}")
    print(f"{'=' * 80}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "done" if success else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status} - {elapsed:.1f}s")
    return success, elapsed


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    env = dict(os.environ)
    env["KARISA_USE_HPA"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print("=" * 80)
    print("  KARISA PROJECT - FULL ANALYSIS PIPELINE (WITH HPA)")
    print(f"  Dataset: {DATASET_PATH.relative_to(ROOT)}")
    print("=" * 80)

    total_start = time.time()
    results = []

    for idx, (label, script) in enumerate(STEPS):
        success, elapsed = run_step(idx, label, script, env)
        results.append((label, success, elapsed))
        if not success:
            print(f"\nPipeline stopped at step {idx + 1} due to failure.")
            break

    total = time.time() - total_start

    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for label, success, elapsed in results:
        mark = "OK" if success else "FAIL"
        print(f"  [{mark}]  {label:<45}  {elapsed:6.1f}s")
    print(f"{'=' * 80}")
    print(f"  Total: {total:.1f}s")
    print(f"{'=' * 80}\n")

    all_ok = all(success for _, success, _ in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
