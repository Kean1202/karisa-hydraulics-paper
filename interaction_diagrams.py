# -*- coding: utf-8 -*-
"""
Interaction Diagrams - Variable Pair Interactions for All Targets

Creates interaction plots showing how the most influential variable pairs
affect each target outcome:
- HYDRAULIC FAILURES (WEEP + FLOOD): Top 5 variables (HDIAM, NHOLES, DIAM, WEIRHT, TRAYSPC)
  Includes decision boundary lines showing class separation regions
- CONVERSION: Top 4 variables (DIAM, WEIRHT, NPASS, TRAYSPC)
- PURITY: Top 4 variables (DIAM, WEIRHT, NPASS, TRAYSPC)

Generates 3 images total, each with 6 subplots (2x3 grid) showing all critical pairs.
Uses colorblind-friendly palettes.

Made with love for Karisa - showing the critical interactions!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    load_data, filter_invalid_values, deduplicate_data, create_binary_targets,
    DESC_COLORS, VARIABLE_LABELS, format_axis_for_paper, convert_to_percentage
)

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'

# For continuous: use magma
CONTINUOUS_CMAP = 'magma'

print("=" * 80)
print("INTENTION DIAGRAMS - Critical Variable Pair Interactions")
print("Generating 3 images, each with 6 interaction plots")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/AmAc_Tray.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)
df_full = create_binary_targets(df_full)

# Convert CONV and PURITY to percentages
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples (CONV & PURITY converted to %)")

# Create output directory
output_dir = Path("results/interaction_diagrams")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Define variable pairs for each target (based on importance rankings)
# ===================================================================

# HYDRAULIC FAILURES (WEEP + FLOOD combined)
# WEEP top 4: HDIAM, NHOLES, DIAM, WEIRHT
# FLOOD top 4: HDIAM, NHOLES, TRAYSPC, DIAM
# Union: HDIAM, NHOLES, DIAM, WEIRHT, TRAYSPC
# HDIAM always on y-axis (second position)
hydraulic_pairs = [
    ('NHOLES', 'HDIAM'),   # Critical for both WEEP (#1,#2) and FLOOD (#1,#2)
    ('DIAM', 'HDIAM'),     # WEEP #3 vs #1, FLOOD #4 vs #1
    ('TRAYSPC', 'HDIAM'),  # FLOOD #3 vs #1
    ('WEIRHT', 'HDIAM'),   # WEEP #4 vs #1
    ('NHOLES', 'DIAM'),    # Important for both
    ('NHOLES', 'TRAYSPC')  # FLOOD #2 vs #3
]

# CONVERSION: Top 4 are DIAM, WEIRHT, NPASS, TRAYSPC
conversion_pairs = [
    ('DIAM', 'WEIRHT'),    # #1 vs #2
    ('DIAM', 'NPASS'),     # #1 vs #3
    ('DIAM', 'TRAYSPC'),   # #1 vs #4
    ('WEIRHT', 'NPASS'),   # #2 vs #3
    ('WEIRHT', 'TRAYSPC'), # #2 vs #4
    ('NPASS', 'TRAYSPC')   # #3 vs #4
]

# PURITY: Same top 4 as CONVERSION (DIAM, WEIRHT, NPASS, TRAYSPC)
purity_pairs = [
    ('DIAM', 'WEIRHT'),    # #1 vs #2
    ('DIAM', 'NPASS'),     # #1 vs #3
    ('DIAM', 'TRAYSPC'),   # #1 vs #4
    ('WEIRHT', 'NPASS'),   # #2 vs #3
    ('WEIRHT', 'TRAYSPC'), # #2 vs #4
    ('NPASS', 'TRAYSPC')   # #3 vs #4
]

# ===================================================================
# HYDRAULIC FAILURES INTERACTION PLOTS (colored by DESC)
# ===================================================================
print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURES INTERACTION PLOTS (WEEP + FLOOD)")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    ax = axes[idx]

    # Use seaborn scatterplot
    sns.scatterplot(
        data=df_full,
        x=var1,
        y=var2,
        hue='DESC',
        palette=DESC_COLORS,
        alpha=0.5,
        s=30,
        ax=ax
    )

    # Format for paper
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'hydraulic_failures_all_interactions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hydraulic_failures_all_interactions.png")
plt.close()

# ===================================================================
# CONVERSION INTERACTION PLOTS (colored by CONV value)
# ===================================================================
print("\n" + "=" * 80)
print("2. CONVERSION INTERACTION PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(conversion_pairs):
    ax = axes[idx]

    # Use seaborn scatterplot like EDA
    scatter = ax.scatter(
        df_pass[var1],
        df_pass[var2],
        c=df_pass['CONV'],
        cmap=CONTINUOUS_CMAP,
        alpha=0.5,
        s=30
    )

    # Add colorbar for each subplot
    cbar = plt.colorbar(scatter, ax=ax)

    # Format for paper
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label='Conversion (%)', cbar=cbar)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'conversion_all_interactions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: conversion_all_interactions.png")
plt.close()

# ===================================================================
# PURITY INTERACTION PLOTS (colored by PURITY value)
# ===================================================================
print("\n" + "=" * 80)
print("3. PURITY INTERACTION PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(purity_pairs):
    ax = axes[idx]

    # Use same approach as EDA
    scatter = ax.scatter(
        df_pass[var1],
        df_pass[var2],
        c=df_pass['PURITY'],
        cmap=CONTINUOUS_CMAP,
        alpha=0.5,
        s=30
    )

    # Add colorbar for each subplot
    cbar = plt.colorbar(scatter, ax=ax)

    # Format for paper
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label='Purity (%)', cbar=cbar)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'purity_all_interactions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: purity_all_interactions.png")
plt.close()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("INTENTION DIAGRAMS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("\nGenerated 3 images (each with 6 subplots):")
print("  1. hydraulic_failures_all_interactions.png")
print("  2. conversion_all_interactions.png")
print("  3. purity_all_interactions.png")
print("\nVariable pairs per target:")
print(f"\n  HYDRAULIC FAILURES (WEEP + FLOOD): {len(hydraulic_pairs)} pairs from top 5 variables")
print("     (HDIAM, NHOLES, DIAM, WEIRHT, TRAYSPC)")
print(f"  CONVERSION: {len(conversion_pairs)} pairs from top 4 variables")
print("     (DIAM, WEIRHT, NPASS, TRAYSPC)")
print(f"  PURITY: {len(purity_pairs)} pairs from top 4 variables")
print("     (DIAM, WEIRHT, NPASS, TRAYSPC)")
print("\n🎨 Colorblind-friendly interaction diagrams created for Karisa! 🎨")
print("=" * 80)
