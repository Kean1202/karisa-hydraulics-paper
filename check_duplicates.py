# -*- coding: utf-8 -*-
"""
Check Duplicates - Simple Script to Check for Duplicate Rows

Checks both datasets for:
1. Complete duplicate rows (all columns)
2. Duplicate combinations of independent variables

Checks both data files: karisa_paper.xlsx and AmAc_Tray.xlsx

Made with love for Karisa!
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("DUPLICATE CHECK - BOTH DATA FILES")
print("=" * 80)

# Independent variables
independent_vars = ['NHOLES', 'HDIAM', 'TRAYSPC', 'WEIRHT', 'DECK', 'DIAM', 'NPASS']

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

    print(f"\nFull dataset: {df_full.shape}")
    print(f"Pass dataset: {df_pass.shape}")

    # ===================================================================
    # FULL DATASET
    # ===================================================================
    print("\n" + "=" * 80)
    print("FULL DATASET - DUPLICATE CHECK")
    print("=" * 80)

    # Complete duplicates (all columns)
    total_rows = len(df_full)
    duplicate_rows = df_full.duplicated().sum()
    unique_rows = total_rows - duplicate_rows

    print(f"\n1. Complete Row Duplicates (all columns):")
    print(f"   Total rows: {total_rows}")
    print(f"   Duplicate rows: {duplicate_rows}")
    print(f"   Unique rows: {unique_rows}")

    if duplicate_rows > 0:
        print(f"\n   Example duplicates:")
        dup_mask = df_full.duplicated(keep=False)
        print(df_full[dup_mask].head(10))

    # Duplicates based on independent variables only
    indep_duplicates = df_full[independent_vars].duplicated().sum()
    unique_combinations = len(df_full[independent_vars].drop_duplicates())

    print(f"\n2. Duplicate Combinations (independent variables only):")
    print(f"   Total rows: {total_rows}")
    print(f"   Duplicate combinations: {indep_duplicates}")
    print(f"   Unique combinations: {unique_combinations}")
    print(f"   Average replicates per combination: {total_rows / unique_combinations:.1f}")

    # Show distribution of replicates
    combo_counts = df_full[independent_vars].value_counts()
    print(f"\n   Replicate statistics:")
    print(f"   Min replicates: {combo_counts.min()}")
    print(f"   Max replicates: {combo_counts.max()}")
    print(f"   Median replicates: {combo_counts.median():.0f}")
    print(f"   Mean replicates: {combo_counts.mean():.1f}")

    # ===================================================================
    # PASS DATASET
    # ===================================================================
    print("\n" + "=" * 80)
    print("PASS DATASET - DUPLICATE CHECK")
    print("=" * 80)

    # Complete duplicates (all columns)
    total_rows_pass = len(df_pass)
    duplicate_rows_pass = df_pass.duplicated().sum()
    unique_rows_pass = total_rows_pass - duplicate_rows_pass

    print(f"\n1. Complete Row Duplicates (all columns):")
    print(f"   Total rows: {total_rows_pass}")
    print(f"   Duplicate rows: {duplicate_rows_pass}")
    print(f"   Unique rows: {unique_rows_pass}")

    if duplicate_rows_pass > 0:
        print(f"\n   Example duplicates:")
        dup_mask_pass = df_pass.duplicated(keep=False)
        print(df_pass[dup_mask_pass].head(10))

    # Duplicates based on independent variables only
    indep_duplicates_pass = df_pass[independent_vars].duplicated().sum()
    unique_combinations_pass = len(df_pass[independent_vars].drop_duplicates())

    print(f"\n2. Duplicate Combinations (independent variables only):")
    print(f"   Total rows: {total_rows_pass}")
    print(f"   Duplicate combinations: {indep_duplicates_pass}")
    print(f"   Unique combinations: {unique_combinations_pass}")
    print(f"   Average replicates per combination: {total_rows_pass / unique_combinations_pass:.1f}")

    # Show distribution of replicates
    combo_counts_pass = df_pass[independent_vars].value_counts()
    print(f"\n   Replicate statistics:")
    print(f"   Min replicates: {combo_counts_pass.min()}")
    print(f"   Max replicates: {combo_counts_pass.max()}")
    print(f"   Median replicates: {combo_counts_pass.median():.0f}")
    print(f"   Mean replicates: {combo_counts_pass.mean():.1f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nFull Dataset:")
    print(f"  - This is a designed experiment with replicates")
    print(f"  - {unique_combinations} unique combinations × ~{total_rows / unique_combinations:.0f} replicates = {total_rows} total rows")

    print(f"\nPass Dataset:")
    print(f"  - Subset of full dataset (only PASS outcomes)")
    print(f"  - {unique_combinations_pass} unique combinations × ~{total_rows_pass / unique_combinations_pass:.0f} replicates = {total_rows_pass} total rows")

print("\n" + "=" * 80)
print("✓ Duplicate check complete for both files!")
print("=" * 80)
