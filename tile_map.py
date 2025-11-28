# -*- coding: utf-8 -*-
"""
Tile Map Visualizations - Discrete Grid Heatmaps for Variable Interactions

Creates tile/heatmap visualizations where each unique combination of two variables
gets a colored square based on the outcome. NO interpolation, NO smoothing.
Colors change only at discrete grid boundaries.

Uses the same variable pairs as intention_diagrams.py:
- HYDRAULIC FAILURES (WEEP + FLOOD): Top 5 variables
- CONVERSION: Top 4 variables
- PURITY: Top 4 variables

Each tile represents the aggregated outcome for that specific combination.

Made with love for Karisa - clean discrete tiles!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import load_data, filter_invalid_values, create_binary_targets

# Set up plotting style
sns.set_style("white")

print("=" * 80)
print("TILE MAP VISUALIZATIONS - Discrete Grid Heatmaps")
print("Generating 3 images, each with 6 tile maps")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/karisa_paper.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full = create_binary_targets(df_full)

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples")

# Create output directory
output_dir = Path("results/tile_maps")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Define variable pairs (same as intention_diagrams.py)
# ===================================================================

# HYDRAULIC FAILURES (WEEP + FLOOD combined)
hydraulic_pairs = [
    ('NHOLES', 'HDIAM'),
    ('DIAM', 'HDIAM'),
    ('TRAYSPC', 'HDIAM'),
    ('WEIRHT', 'HDIAM'),
    ('NHOLES', 'DIAM'),
    ('NHOLES', 'TRAYSPC')
]

# CONVERSION
conversion_pairs = [
    ('DIAM', 'WEIRHT'),
    ('DIAM', 'NPASS'),
    ('DIAM', 'TRAYSPC'),
    ('WEIRHT', 'NPASS'),
    ('WEIRHT', 'TRAYSPC'),
    ('NPASS', 'TRAYSPC')
]

# PURITY (same pairs as CONVERSION)
purity_pairs = [
    ('DIAM', 'WEIRHT'),
    ('DIAM', 'NPASS'),
    ('DIAM', 'TRAYSPC'),
    ('WEIRHT', 'NPASS'),
    ('WEIRHT', 'TRAYSPC'),
    ('NPASS', 'TRAYSPC')
]

# ===================================================================
# HYDRAULIC FAILURES TILE MAPS
# Calculate % of failures (WEEP + FLOOD) for each combination
# ===================================================================
print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURES TILE MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    if var1 in df_full.columns and var2 in df_full.columns:
        ax = axes[idx]

        # Calculate failure rate for each combination
        # Group by both variables and calculate % of failures (WEEP or FLOOD)
        df_full['is_failure'] = (df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')
        grouped = df_full.groupby([var1, var2])['is_failure'].mean() * 100  # Convert to percentage

        # Create pivot table for heatmap
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='is_failure')

        # Plot using pcolormesh (no interpolation)
        # Get unique sorted values for proper positioning
        x_vals = sorted(df_full[var1].unique())
        y_vals = sorted(df_full[var2].unique())

        # Create mesh grid edges (for pcolormesh)
        x_edges = np.arange(len(x_vals) + 1)
        y_edges = np.arange(len(y_vals) + 1)

        # Use pcolormesh with no interpolation
        im = ax.pcolormesh(x_edges, y_edges, pivot.values,
                          cmap='RdYlGn_r', vmin=0, vmax=100,
                          shading='flat', edgecolors='white', linewidth=0.5)

        # Set ticks to center of each cell
        ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        ax.set_yticks(np.arange(len(y_vals)) + 0.5)
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nFailure Rate (%)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Failure %', fontsize=10)

fig.suptitle('Hydraulic Failures: Tile Map Visualization\n(% WEEP + FLOOD per combination)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'hydraulic_failures_tile_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hydraulic_failures_tile_maps.png")
plt.close()

# Clean up temporary column
df_full.drop('is_failure', axis=1, inplace=True)

# ===================================================================
# CONVERSION TILE MAPS
# Calculate mean CONVERSION for each combination
# ===================================================================
print("\n" + "=" * 80)
print("2. CONVERSION TILE MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(conversion_pairs):
    if var1 in df_pass.columns and var2 in df_pass.columns:
        ax = axes[idx]

        # Calculate mean CONVERSION for each combination
        grouped = df_pass.groupby([var1, var2])['CONV'].mean()

        # Create pivot table for heatmap
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='CONV')

        # Get unique sorted values
        x_vals = sorted(df_pass[var1].unique())
        y_vals = sorted(df_pass[var2].unique())

        # Create mesh grid edges
        x_edges = np.arange(len(x_vals) + 1)
        y_edges = np.arange(len(y_vals) + 1)

        # Use pcolormesh with no interpolation
        im = ax.pcolormesh(x_edges, y_edges, pivot.values,
                          cmap='viridis',
                          shading='flat', edgecolors='white', linewidth=0.5)

        # Set ticks to center of each cell
        ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        ax.set_yticks(np.arange(len(y_vals)) + 0.5)
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nMean CONVERSION', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CONVERSION', fontsize=10)

fig.suptitle('CONVERSION: Tile Map Visualization\n(Mean CONVERSION per combination)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'conversion_tile_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: conversion_tile_maps.png")
plt.close()

# ===================================================================
# PURITY TILE MAPS
# Calculate mean PURITY for each combination
# ===================================================================
print("\n" + "=" * 80)
print("3. PURITY TILE MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(purity_pairs):
    if var1 in df_pass.columns and var2 in df_pass.columns:
        ax = axes[idx]

        # Calculate mean PURITY for each combination
        grouped = df_pass.groupby([var1, var2])['PURITY'].mean()

        # Create pivot table for heatmap
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='PURITY')

        # Get unique sorted values
        x_vals = sorted(df_pass[var1].unique())
        y_vals = sorted(df_pass[var2].unique())

        # Create mesh grid edges
        x_edges = np.arange(len(x_vals) + 1)
        y_edges = np.arange(len(y_vals) + 1)

        # Use pcolormesh with no interpolation
        im = ax.pcolormesh(x_edges, y_edges, pivot.values,
                          cmap='viridis',
                          shading='flat', edgecolors='white', linewidth=0.5)

        # Set ticks to center of each cell
        ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        ax.set_yticks(np.arange(len(y_vals)) + 0.5)
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nMean PURITY', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PURITY', fontsize=10)

fig.suptitle('PURITY: Tile Map Visualization\n(Mean PURITY per combination)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'purity_tile_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: purity_tile_maps.png")
plt.close()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("TILE MAP VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("\nGenerated 3 images (each with 6 tile maps):")
print("  1. hydraulic_failures_tile_maps.png")
print("  2. conversion_tile_maps.png")
print("  3. purity_tile_maps.png")
print("\nVisualization details:")
print("  - NO interpolation (discrete tiles only)")
print("  - NO smoothing (sharp boundaries)")
print("  - Equal aspect ratio (square tiles)")
print("  - White gridlines between tiles")
print("  - Aggregated outcomes per unique combination")
print("\n🟦 Discrete tile maps created for Karisa! 🟦")
print("=" * 80)
