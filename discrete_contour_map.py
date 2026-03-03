# -*- coding: utf-8 -*-
"""
Discrete Contour Map Visualizations - Contour Maps Based on Discrete Grid

Creates contour map visualizations where contour lines are based ONLY on the
existing discrete grid. NO artificial smoothing, NO interpolation beyond the grid.

Uses the same variable pairs as intention_diagrams.py and tile_map.py:
- HYDRAULIC FAILURES (WEEP + FLOOD): Top 5 variables
- CONVERSION: Top 4 variables
- PURITY: Top 4 variables

Contour lines show zones/regions, but are based purely on discrete data.
Light gray grid overlay shows the actual discrete tiles.

Made with love for Karisa - discrete contours with grid overlay!
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
    VARIABLE_LABELS, format_axis_for_paper, convert_to_percentage
)

# Set up plotting style
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'

print("=" * 80)
print("DISCRETE CONTOUR MAP VISUALIZATIONS")
print("Generating 3 images, each with 6 contour maps")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/AmAc_Tray.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
# df_full, df_pass = deduplicate_data(df_full, df_pass)  # REMOVED - no duplicates in data
df_full = create_binary_targets(df_full)

# Convert CONV and PURITY to percentages
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples (CONV & PURITY in %)")

# Create output directory
output_dir = Path("results/discrete_contour_maps")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Define variable pairs (same as tile_map.py)
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
# HYDRAULIC FAILURES CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURES CONTOUR MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    if var1 in df_full.columns and var2 in df_full.columns:
        ax = axes[idx]

        # Calculate failure rate for each combination
        df_full['is_failure'] = (df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')
        grouped = df_full.groupby([var1, var2])['is_failure'].mean() * 100

        # Create pivot table - discrete grid
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='is_failure')

        # Get actual discrete values
        x_vals = sorted(df_full[var1].unique())
        y_vals = sorted(df_full[var2].unique())

        # Create coordinate arrays (centered at discrete points)
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

        # Define contour levels based on quantiles (discrete thresholds)
        values = pivot.values.flatten()
        values = values[~np.isnan(values)]
        levels = [0, 25, 50, 75, 100]  # Fixed thresholds for failure %

        # Filled contour plot (no interpolation beyond grid)
        contourf = ax.contourf(X, Y, pivot.values, levels=levels,
                               cmap='magma', alpha=0.8, extend='neither')

        # Contour lines
        contour = ax.contour(X, Y, pivot.values, levels=levels,
                            colors='white', linewidths=1.5, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f%%')

        # Overlay discrete grid (light gray boundaries)
        x_edges = np.arange(len(x_vals) + 1) - 0.5
        y_edges = np.arange(len(y_vals) + 1) - 0.5
        for x_edge in x_edges:
            ax.axvline(x_edge, color='lightgray', linewidth=0.5, alpha=0.5)
        for y_edge in y_edges:
            ax.axhline(y_edge, color='lightgray', linewidth=0.5, alpha=0.5)

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)

        # Format for paper
        format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label='Failure Rate (%)', cbar=cbar)
        ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(output_dir / 'hydraulic_failures_contour_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hydraulic_failures_contour_maps.png")
plt.close()

# Clean up temporary column
df_full.drop('is_failure', axis=1, inplace=True)

# ===================================================================
# CONVERSION CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("2. CONVERSION CONTOUR MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(conversion_pairs):
    if var1 in df_pass.columns and var2 in df_pass.columns:
        ax = axes[idx]

        # Calculate mean CONVERSION for each combination
        grouped = df_pass.groupby([var1, var2])['CONV'].mean()

        # Create pivot table - discrete grid
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='CONV')

        # Get actual discrete values
        x_vals = sorted(df_pass[var1].unique())
        y_vals = sorted(df_pass[var2].unique())

        # Create coordinate arrays
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

        # Define contour levels based on quantiles
        values = pivot.values.flatten()
        values = values[~np.isnan(values)]
        levels = np.percentile(values, [0, 25, 50, 75, 100])

        # Filled contour plot
        contourf = ax.contourf(X, Y, pivot.values, levels=levels,
                               cmap='magma', alpha=0.8, extend='neither')

        # Contour lines
        contour = ax.contour(X, Y, pivot.values, levels=levels,
                            colors='white', linewidths=1.5, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.4f')

        # Overlay discrete grid
        x_edges = np.arange(len(x_vals) + 1) - 0.5
        y_edges = np.arange(len(y_vals) + 1) - 0.5
        for x_edge in x_edges:
            ax.axvline(x_edge, color='lightgray', linewidth=0.5, alpha=0.5)
        for y_edge in y_edges:
            ax.axhline(y_edge, color='lightgray', linewidth=0.5, alpha=0.5)

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)

        # Format for paper
        format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label='Conversion (%)', cbar=cbar)
        ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(output_dir / 'conversion_contour_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: conversion_contour_maps.png")
plt.close()

# ===================================================================
# PURITY CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("3. PURITY CONTOUR MAPS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(purity_pairs):
    if var1 in df_pass.columns and var2 in df_pass.columns:
        ax = axes[idx]

        # Calculate mean PURITY for each combination
        grouped = df_pass.groupby([var1, var2])['PURITY'].mean()

        # Create pivot table - discrete grid
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='PURITY')

        # Get actual discrete values
        x_vals = sorted(df_pass[var1].unique())
        y_vals = sorted(df_pass[var2].unique())

        # Create coordinate arrays
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

        # Define contour levels based on quantiles
        values = pivot.values.flatten()
        values = values[~np.isnan(values)]
        levels = np.percentile(values, [0, 25, 50, 75, 100])

        # Filled contour plot
        contourf = ax.contourf(X, Y, pivot.values, levels=levels,
                               cmap='magma', alpha=0.8, extend='neither')

        # Contour lines
        contour = ax.contour(X, Y, pivot.values, levels=levels,
                            colors='white', linewidths=1.5, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.4f')

        # Overlay discrete grid
        x_edges = np.arange(len(x_vals) + 1) - 0.5
        y_edges = np.arange(len(y_vals) + 1) - 0.5
        for x_edge in x_edges:
            ax.axvline(x_edge, color='lightgray', linewidth=0.5, alpha=0.5)
        for y_edge in y_edges:
            ax.axhline(y_edge, color='lightgray', linewidth=0.5, alpha=0.5)

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)

        # Format for paper
        format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label='Purity (%)', cbar=cbar)
        ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(output_dir / 'purity_contour_maps.png', dpi=300, bbox_inches='tight')
print("✓ Saved: purity_contour_maps.png")
plt.close()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("DISCRETE CONTOUR MAP VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("\nGenerated 3 images (each with 6 contour maps):")
print("  1. hydraulic_failures_contour_maps.png")
print("  2. conversion_contour_maps.png")
print("  3. purity_contour_maps.png")
print("\nVisualization details:")
print("  - Contour lines based ONLY on discrete grid")
print("  - NO artificial smoothing or interpolation")
print("  - Light gray grid overlay shows discrete tiles")
print("  - Quantile-based contour levels")
print("  - Equal aspect ratio")
print("\n📈 Discrete contour maps created for Karisa! 📈")
print("=" * 80)
