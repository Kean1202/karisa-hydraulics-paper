# -*- coding: utf-8 -*-
"""
Check Duplicate Consistency - Analyze Consistency of Duplicates

For rows with duplicate independent variable combinations:
1. Check if DESC values are consistent
2. Show differences in PURITY values
3. Save all duplicates to Excel showing each duplicate group

Checks both data files: karisa_paper.xlsx and AmAc_Tray.xlsx

Made with love for Karisa!
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("DUPLICATE CONSISTENCY CHECK - BOTH DATA FILES")
print("=" * 80)

# Independent variables
independent_vars = ['NHOLES', 'HDIAM', 'TRAYSPC', 'WEIRHT', 'DECK', 'DIAM', 'NPASS']

# Data files to check
data_files = [
    Path("data/karisa_paper.xlsx"),
    Path("data/AmAc_Tray.xlsx")
]

# Create output directory
output_dir = Path("results/duplicate_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

for data_path in data_files:
    print("\n" + "=" * 80)
    print(f"DATA FILE: {data_path.name}")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df_full = pd.read_excel(data_path, sheet_name="full_dataset")
    df_pass = pd.read_excel(data_path, sheet_name="pass_only")

    print(f"Full dataset: {df_full.shape[0]} rows")
    print(f"Pass dataset: {df_pass.shape[0]} rows")

    # Show columns
    print(f"Columns in full dataset: {list(df_full.columns)}")
    print(f"Independent variables used for grouping: {independent_vars}")
    if 'NO' in df_full.columns:
        print("Note: 'NO' column will be excluded from duplicate checking (it's just a row number)")

    # ===================================================================
    # FULL DATASET ANALYSIS
    # ===================================================================
    print("\n" + "=" * 80)
    print("FULL DATASET - DUPLICATE CONSISTENCY")
    print("=" * 80)

    # Create a group identifier for each unique combination
    df_full['combo_id'] = df_full.groupby(independent_vars).ngroup()

    # Find combinations that have duplicates
    combo_counts = df_full['combo_id'].value_counts()
    duplicate_combos = combo_counts[combo_counts > 1]

    print(f"\nTotal unique combinations: {len(combo_counts)}")
    print(f"Combinations with duplicates: {len(duplicate_combos)}")

    if len(duplicate_combos) == 0:
        print("\n✓ No duplicates found in full dataset - all combinations are unique!")
        # Still need to check pass dataset, so don't continue yet
    else:
        print(f"Total duplicate rows: {duplicate_combos.sum()}")

        # Analyze DESC consistency
        print("\n" + "-" * 80)
        print("DESC CONSISTENCY CHECK")
        print("-" * 80)

        desc_inconsistent = []
        for combo_id in duplicate_combos.index:
            group = df_full[df_full['combo_id'] == combo_id]
            unique_desc = group['DESC'].unique()

            if len(unique_desc) > 1:
                desc_inconsistent.append({
                    'combo_id': combo_id,
                    'count': len(group),
                    'desc_values': unique_desc.tolist(),
                    'desc_counts': group['DESC'].value_counts().to_dict()
                })

        if len(desc_inconsistent) > 0:
            print(f"\n⚠ Found {len(desc_inconsistent)} combinations with INCONSISTENT DESC values:")
            for item in desc_inconsistent[:10]:  # Show first 10
                print(f"\n  Combo ID {item['combo_id']}:")
                print(f"    Total rows: {item['count']}")
                print(f"    DESC values found: {item['desc_values']}")
                print(f"    Distribution: {item['desc_counts']}")

            if len(desc_inconsistent) > 10:
                print(f"\n  ... and {len(desc_inconsistent) - 10} more")
        else:
            print("\n✓ All duplicate combinations have CONSISTENT DESC values")

        # Create detailed duplicate report for full dataset
        duplicate_rows_full = df_full[df_full['combo_id'].isin(duplicate_combos.index)].copy()
        duplicate_rows_full = duplicate_rows_full.sort_values(['combo_id'] + independent_vars)

        # Add statistics for each group
        duplicate_rows_full['group_size'] = duplicate_rows_full.groupby('combo_id')['combo_id'].transform('count')
        duplicate_rows_full['row_in_group'] = duplicate_rows_full.groupby('combo_id').cumcount() + 1

        print(f"\nSaving full dataset duplicates to Excel...")
        excel_filename = output_dir / f'duplicates_{data_path.stem}_full.xlsx'

        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Main sheet with all duplicates
            duplicate_rows_full.to_excel(writer, sheet_name='All_Duplicates', index=False)

            # Summary sheet
            summary_df = pd.DataFrame({
                'combo_id': duplicate_combos.index,
                'duplicate_count': duplicate_combos.values
            })
            summary_df = summary_df.sort_values('duplicate_count', ascending=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # DESC inconsistency sheet if applicable
            if len(desc_inconsistent) > 0:
                desc_df = pd.DataFrame(desc_inconsistent)
                desc_df.to_excel(writer, sheet_name='DESC_Inconsistencies', index=False)

        print(f"✓ Saved: {excel_filename}")

    # ===================================================================
    # PASS DATASET ANALYSIS
    # ===================================================================
    print("\n" + "=" * 80)
    print("PASS DATASET - DUPLICATE CONSISTENCY & PURITY DIFFERENCES")
    print("=" * 80)

    # Create a group identifier for each unique combination
    df_pass['combo_id'] = df_pass.groupby(independent_vars).ngroup()

    # Find combinations that have duplicates
    combo_counts_pass = df_pass['combo_id'].value_counts()
    duplicate_combos_pass = combo_counts_pass[combo_counts_pass > 1]

    print(f"\nTotal unique combinations: {len(combo_counts_pass)}")
    print(f"Combinations with duplicates: {len(duplicate_combos_pass)}")

    if len(duplicate_combos_pass) == 0:
        print("\n✓ No duplicates found in pass dataset - all combinations are unique!")
        # Clean up temporary columns before skipping
        df_full.drop('combo_id', axis=1, inplace=True, errors='ignore')
        df_pass.drop('combo_id', axis=1, inplace=True, errors='ignore')
        continue  # Skip to next file

    print(f"Total duplicate rows: {duplicate_combos_pass.sum()}")

    # Analyze PURITY differences
    print("\n" + "-" * 80)
    print("PURITY DIFFERENCE ANALYSIS")
    print("-" * 80)

    purity_stats = []
    for combo_id in duplicate_combos_pass.index:
        group = df_pass[df_pass['combo_id'] == combo_id]
        purity_values = group['PURITY'].values

        purity_stats.append({
            'combo_id': combo_id,
            'count': len(group),
            'purity_min': purity_values.min(),
            'purity_max': purity_values.max(),
            'purity_mean': purity_values.mean(),
            'purity_std': purity_values.std(),
            'purity_range': purity_values.max() - purity_values.min()
        })

    purity_stats_df = pd.DataFrame(purity_stats)
    purity_stats_df = purity_stats_df.sort_values('purity_range', ascending=False)

    print(f"\nPURITY Statistics for Duplicates:")
    print(f"  Mean purity range: {purity_stats_df['purity_range'].mean():.6f}")
    print(f"  Max purity range: {purity_stats_df['purity_range'].max():.6f}")
    print(f"  Min purity range: {purity_stats_df['purity_range'].min():.6f}")
    print(f"  Std of purity range: {purity_stats_df['purity_range'].std():.6f}")

    print(f"\nTop 10 combinations with largest PURITY variation:")
    print(purity_stats_df.head(10).to_string(index=False))

    # Analyze CONVERSION differences too
    print("\n" + "-" * 80)
    print("CONVERSION DIFFERENCE ANALYSIS")
    print("-" * 80)

    conv_stats = []
    for combo_id in duplicate_combos_pass.index:
        group = df_pass[df_pass['combo_id'] == combo_id]
        conv_values = group['CONV'].values

        conv_stats.append({
            'combo_id': combo_id,
            'count': len(group),
            'conv_min': conv_values.min(),
            'conv_max': conv_values.max(),
            'conv_mean': conv_values.mean(),
            'conv_std': conv_values.std(),
            'conv_range': conv_values.max() - conv_values.min()
        })

    conv_stats_df = pd.DataFrame(conv_stats)
    conv_stats_df = conv_stats_df.sort_values('conv_range', ascending=False)

    print(f"\nCONVERSION Statistics for Duplicates:")
    print(f"  Mean conversion range: {conv_stats_df['conv_range'].mean():.6f}")
    print(f"  Max conversion range: {conv_stats_df['conv_range'].max():.6f}")
    print(f"  Min conversion range: {conv_stats_df['conv_range'].min():.6f}")
    print(f"  Std of conversion range: {conv_stats_df['conv_range'].std():.6f}")

    print(f"\nTop 10 combinations with largest CONVERSION variation:")
    print(conv_stats_df.head(10).to_string(index=False))

    # Create detailed duplicate report for pass dataset
    duplicate_rows_pass = df_pass[df_pass['combo_id'].isin(duplicate_combos_pass.index)].copy()
    duplicate_rows_pass = duplicate_rows_pass.sort_values(['combo_id'] + independent_vars)

    # Add statistics for each group
    duplicate_rows_pass['group_size'] = duplicate_rows_pass.groupby('combo_id')['combo_id'].transform('count')
    duplicate_rows_pass['row_in_group'] = duplicate_rows_pass.groupby('combo_id').cumcount() + 1

    # Add group-level statistics
    duplicate_rows_pass['group_purity_mean'] = duplicate_rows_pass.groupby('combo_id')['PURITY'].transform('mean')
    duplicate_rows_pass['group_purity_std'] = duplicate_rows_pass.groupby('combo_id')['PURITY'].transform('std')
    duplicate_rows_pass['group_purity_range'] = duplicate_rows_pass.groupby('combo_id')['PURITY'].transform(lambda x: x.max() - x.min())

    duplicate_rows_pass['group_conv_mean'] = duplicate_rows_pass.groupby('combo_id')['CONV'].transform('mean')
    duplicate_rows_pass['group_conv_std'] = duplicate_rows_pass.groupby('combo_id')['CONV'].transform('std')
    duplicate_rows_pass['group_conv_range'] = duplicate_rows_pass.groupby('combo_id')['CONV'].transform(lambda x: x.max() - x.min())

    print(f"\nSaving pass dataset duplicates to Excel...")
    excel_filename_pass = output_dir / f'duplicates_{data_path.stem}_pass.xlsx'

    with pd.ExcelWriter(excel_filename_pass, engine='openpyxl') as writer:
        # Main sheet with all duplicates
        duplicate_rows_pass.to_excel(writer, sheet_name='All_Duplicates', index=False)

        # Summary sheet
        summary_pass_df = pd.DataFrame({
            'combo_id': duplicate_combos_pass.index,
            'duplicate_count': duplicate_combos_pass.values
        })
        summary_pass_df = summary_pass_df.sort_values('duplicate_count', ascending=False)
        summary_pass_df.to_excel(writer, sheet_name='Summary', index=False)

        # PURITY statistics
        purity_stats_df.to_excel(writer, sheet_name='PURITY_Stats', index=False)

        # CONVERSION statistics
        conv_stats_df.to_excel(writer, sheet_name='CONV_Stats', index=False)

    print(f"✓ Saved: {excel_filename_pass}")

    # Clean up temporary columns
    df_full.drop('combo_id', axis=1, inplace=True, errors='ignore')
    df_pass.drop('combo_id', axis=1, inplace=True, errors='ignore')

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nAll duplicate analysis files saved to: {output_dir.absolute()}")
print("\nFiles created:")
print("  For each data file:")
print("    - duplicates_[filename]_full.xlsx (full dataset duplicates)")
print("    - duplicates_[filename]_pass.xlsx (pass dataset duplicates with PURITY/CONV analysis)")
print("\n✓ Duplicate consistency check complete for both files!")
print("=" * 80)
