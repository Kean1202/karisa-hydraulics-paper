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

from xgboost import XGBClassifier

# Import utilities
from utils import load_data, filter_invalid_values, deduplicate_data, create_binary_targets

# Set up plotting style
sns.set_style("whitegrid")

# Colorblind-friendly palettes
# For categorical DESC: use colorblind-safe colors
DESC_COLORS = {
    'PASS': '#0173B2',    # Blue
    'WEEP': '#DE8F05',    # Orange
    'FLOOD': '#CC78BC'    # Purple/Pink
}

# For continuous: viridis is already colorblind-friendly
CONTINUOUS_CMAP = 'viridis'

print("=" * 80)
print("INTENTION DIAGRAMS - Critical Variable Pair Interactions")
print("Generating 3 images, each with 6 interaction plots")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/karisa_paper.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)
df_full = create_binary_targets(df_full)

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples")

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

    # Add decision boundary lines
    # Train XGBoost on just these 2 variables
    X_pair = df_full[[var1, var2]].values
    y_desc = df_full['DESC'].map({'PASS': 0, 'WEEP': 1, 'FLOOD': 2}).values

    clf = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    clf.fit(X_pair, y_desc)

    # Create meshgrid for decision boundaries
    x_min, x_max = X_pair[:, 0].min(), X_pair[:, 0].max()
    y_min, y_max = X_pair[:, 1].min(), X_pair[:, 1].max()

    # Add small padding
    x_padding = (x_max - x_min) * 0.02
    y_padding = (y_max - y_min) * 0.02

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_padding, x_max + x_padding, 200),
        np.linspace(y_min - y_padding, y_max + y_padding, 200)
    )

    # Predict on meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Draw decision boundaries with colored lines
    # Contour lines at class boundaries (where prediction changes)
    # Use DESC_COLORS: WEEP orange for PASS/WEEP boundary, FLOOD purple for WEEP/FLOOD boundary
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors=[DESC_COLORS['WEEP'], DESC_COLORS['FLOOD']],
               linewidths=0.8, linestyles='--', alpha=0.6)

    ax.set_xlabel(var1, fontsize=11, fontweight='bold')
    ax.set_ylabel(var2, fontsize=11, fontweight='bold')
    ax.set_title(f'{var1} vs {var2}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Hydraulic Failures: Critical Variable Pair Interactions\n(colored by DESC outcome)',
             fontsize=16, fontweight='bold', y=0.995)
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
    cbar.set_label('CONV', fontsize=9)

    ax.set_xlabel(var1, fontsize=11, fontweight='bold')
    ax.set_ylabel(var2, fontsize=11, fontweight='bold')
    ax.set_title(f'{var1} vs {var2}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

fig.suptitle('CONVERSION: Critical Variable Pair Interactions\n(colored by CONVERSION value)',
             fontsize=16, fontweight='bold', y=0.995)
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
    cbar.set_label('PURITY', fontsize=9)

    ax.set_xlabel(var1, fontsize=11, fontweight='bold')
    ax.set_ylabel(var2, fontsize=11, fontweight='bold')
    ax.set_title(f'{var1} vs {var2}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

fig.suptitle('PURITY: Critical Variable Pair Interactions\n(colored by PURITY value)',
             fontsize=16, fontweight='bold', y=0.995)
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
