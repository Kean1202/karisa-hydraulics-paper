# -*- coding: utf-8 -*-
"""
Discrete Tile Maps with Contour Lines for Hydraulic Failures

Generates discrete tile maps showing empirical failure rate per factorial
combination, with contour lines overlaid to highlight regions of similar
failure rates. NO interpolation between non-existent combinations.

Each tile represents one actual experimental combination. Contour lines
suggest regional patterns across the discrete grid.

Made with love for Karisa - discrete tiles with contours!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import load_data, filter_invalid_values

# Set up plotting style
sns.set_style("white")

print("=" * 80)
print("DISCRETE TILE MAPS WITH CONTOUR LINES")
print("Empirical failure rates on factorial grid")
print("=" * 80)

# ==================================================================
# 1. LOAD AND PREPARE DATA
# ==================================================================
print("\n1. Loading and preparing data...")
df_full, df_pass = load_data(data_path="data/karisa_paper.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)

# Create binary failure target
df_full['is_failure'] = ((df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')).astype(int)

print(f"   Full dataset: {len(df_full)} samples")
print(f"   Overall failure rate: {df_full['is_failure'].mean() * 100:.1f}%")

# ==================================================================
# 2. CREATE TILE MAPS FOR EACH VARIABLE PAIR
# ==================================================================
print("\n2. Creating discrete tile maps with contours...")

# Define variable pairs
hydraulic_pairs = [
    ('NHOLES', 'HDIAM'),
    ('DIAM', 'HDIAM'),
    ('TRAYSPC', 'HDIAM'),
    ('WEIRHT', 'HDIAM'),
    ('NHOLES', 'DIAM'),
    ('NHOLES', 'TRAYSPC')
]

# Create output directory
output_dir = Path("results/hydraulics")
output_dir.mkdir(parents=True, exist_ok=True)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    print(f"   Processing {var1} vs {var2}...")
    ax = axes[idx]

    # Calculate empirical failure rate for each discrete combination
    grouped = df_full.groupby([var1, var2])['is_failure'].mean()

    # Pivot to 2D matrix
    pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='is_failure')

    # Get discrete levels
    x_vals = sorted(df_full[var1].unique())
    y_vals = sorted(df_full[var2].unique())

    # Create mesh for tiles (edges for pcolormesh)
    # Need one extra point for edges when using shading='flat'
    X_edges, Y_edges = np.meshgrid(np.arange(len(x_vals) + 1) - 0.5,
                                     np.arange(len(y_vals) + 1) - 0.5)

    # Create mesh for contours (centers)
    X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

    # Tile map using pcolormesh
    im = ax.pcolormesh(X_edges, Y_edges, pivot.values,
                       cmap='magma', vmin=0, vmax=1,
                       shading='flat', edgecolors='white', linewidth=0.8)

    # Contour lines at 0.25, 0.5, 0.75
    contour = ax.contour(X, Y, pivot.values, levels=[0.25, 0.5, 0.75],
                        colors='white', linewidths=1.5, alpha=0.7,
                        linestyles='--')
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Optional gridlines to emphasize discrete structure
    ax.grid(True, alpha=0.3, linewidth=0.5, color='lightgray')

    # Set ticks to actual discrete values
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals],
                       rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals],
                       fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Empirical failure rate', fontsize=10)

    # Labels and formatting
    ax.set_xlabel(var1, fontsize=12, fontweight='bold')
    ax.set_ylabel(var2, fontsize=12, fontweight='bold')
    ax.set_title(f'{var1} vs {var2}\nEmpirical failure rate (tile map)',
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

# Add main title
fig.suptitle('Discrete Tile Maps with Contours for Hydraulic Failures\n(Empirical failure rates per factorial combination)',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = output_dir / 'discrete_tile_contours.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n3. Saved: {output_path}")
plt.close()

print("\n" + "=" * 80)
print("DISCRETE TILE MAP GENERATION COMPLETE!")
print("=" * 80)
print(f"\nPlot saved to: {output_dir.absolute()}")
print("\nVisualization details:")
print("  - Each tile = one factorial combination (NO interpolation)")
print("  - Tile color = empirical failure rate (0-1)")
print("  - White contour lines at 0.25, 0.5, 0.75 failure rates")
print("  - White gridlines emphasize discrete structure")
print("  - Equal aspect ratio (square tiles)")
print("\n🟦 Discrete tile maps created for Karisa! 🟦")
print("=" * 80)
