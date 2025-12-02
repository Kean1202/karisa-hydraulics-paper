# -*- coding: utf-8 -*-
"""
Check Invalid Values - Count Rows with Values Outside Valid Range

Checks both datasets for rows with values outside the defined valid values.
Checks both data files: karisa_paper.xlsx and AmAc_Tray.xlsx

Made with love for Karisa!
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("INVALID VALUES CHECK - BOTH DATA FILES")
print("=" * 80)

# Valid values (whitelist)
VALID_VALUES = {
    'NHOLES': [500, 1625, 2750, 3875, 5000],
    'HDIAM': [0.0025, 0.004875, 0.00725, 0.009625, 0.012],
    'TRAYSPC': [0.5, 0.5625, 0.625, 0.6875, 0.75],
    'WEIRHT': [0.04, 0.0525, 0.065, 0.0775, 0.09],
    'DECK': [1.88, 2.9975, 4.115, 5.2325, 6.35],
    'DIAM': [1, 1.5, 2, 2.5, 3],
    'NPASS': [1, 2, 3, 4]
}

# Data files to check
data_files = [
    Path("data/karisa_paper.xlsx"),
    Path("data/AmAc_Tray.xlsx")
]

for data_path in data_files:
    print("\n" + "=" * 80)
    print(f"DATA FILE: {data_path.name}")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {data_path}")

    df_full = pd.read_excel(data_path, sheet_name="full_dataset")
    df_pass = pd.read_excel(data_path, sheet_name="pass_only")

    print(f"\nFull dataset: {df_full.shape[0]} rows")
    print(f"Pass dataset: {df_pass.shape[0]} rows")

    # ===================================================================
    # FULL DATASET
    # ===================================================================
    print("\n" + "=" * 80)
    print("FULL DATASET - INVALID VALUES")
    print("=" * 80)

    invalid_counts_full = {}
    invalid_rows_full = set()

    for var, valid_vals in VALID_VALUES.items():
        if var in df_full.columns:
            # Find rows with invalid values
            is_invalid = ~df_full[var].isin(valid_vals)
            invalid_count = is_invalid.sum()
            invalid_counts_full[var] = invalid_count

            # Track which rows are invalid
            invalid_rows_full.update(df_full[is_invalid].index.tolist())

            print(f"\n{var}:")
            print(f"  Valid values: {valid_vals}")
            print(f"  Invalid rows: {invalid_count}")

            if invalid_count > 0:
                invalid_values = df_full.loc[is_invalid, var].unique()
                print(f"  Invalid values found: {sorted(invalid_values)}")

    print(f"\n" + "-" * 80)
    print(f"TOTAL UNIQUE ROWS WITH ANY INVALID VALUE: {len(invalid_rows_full)}")
    print(f"Percentage of dataset: {len(invalid_rows_full) / len(df_full) * 100:.2f}%")

    # ===================================================================
    # PASS DATASET
    # ===================================================================
    print("\n" + "=" * 80)
    print("PASS DATASET - INVALID VALUES")
    print("=" * 80)

    invalid_counts_pass = {}
    invalid_rows_pass = set()

    for var, valid_vals in VALID_VALUES.items():
        if var in df_pass.columns:
            # Find rows with invalid values
            is_invalid = ~df_pass[var].isin(valid_vals)
            invalid_count = is_invalid.sum()
            invalid_counts_pass[var] = invalid_count

            # Track which rows are invalid
            invalid_rows_pass.update(df_pass[is_invalid].index.tolist())

            print(f"\n{var}:")
            print(f"  Valid values: {valid_vals}")
            print(f"  Invalid rows: {invalid_count}")

            if invalid_count > 0:
                invalid_values = df_pass.loc[is_invalid, var].unique()
                print(f"  Invalid values found: {sorted(invalid_values)}")

    print(f"\n" + "-" * 80)
    print(f"TOTAL UNIQUE ROWS WITH ANY INVALID VALUE: {len(invalid_rows_pass)}")
    print(f"Percentage of dataset: {len(invalid_rows_pass) / len(df_pass) * 100:.2f}%")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nFull Dataset:")
    print(f"  Total rows: {len(df_full)}")
    print(f"  Rows with invalid values: {len(invalid_rows_full)}")
    print(f"  Clean rows: {len(df_full) - len(invalid_rows_full)}")

    print(f"\nPass Dataset:")
    print(f"  Total rows: {len(df_pass)}")
    print(f"  Rows with invalid values: {len(invalid_rows_pass)}")
    print(f"  Clean rows: {len(df_pass) - len(invalid_rows_pass)}")

    if len(invalid_rows_full) == 0 and len(invalid_rows_pass) == 0:
        print("\n✓ All data is clean! No invalid values found.")
    else:
        print(f"\n⚠ Found invalid values - these would be filtered out by filter_invalid_values()")

print("\n" + "=" * 80)
print("✓ Invalid values check complete for both files!")
print("=" * 80)
