# -*- coding: utf-8 -*-
"""
Discrete Contour Map Visualizations - Contour Maps Based on Discrete Grid

Creates contour map visualizations where contour lines are based ONLY on the
existing discrete grid. NO artificial smoothing, NO interpolation beyond the grid.

Each graph is saved independently (one file per variable pair).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    load_data, filter_invalid_values, create_binary_targets,
    format_axis_for_paper, convert_to_percentage, save_figure_elsevier
)

# Set up plotting style
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'

print("=" * 80)
print("DISCRETE CONTOUR MAP VISUALIZATIONS")
print("Generating individual contour maps (one file per graph)")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/new_data.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full = create_binary_targets(df_full)

# Convert CONV and PURITY to percentages
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples (CONV & PURITY in %)")

# Create output directory
output_dir = Path("results/discrete_contour_maps")
output_dir.mkdir(parents=True, exist_ok=True)


def build_levels(values, fixed_levels=None):
    """Build strictly increasing contour levels."""
    if fixed_levels is not None:
        levels = np.array(fixed_levels, dtype=float)
    else:
        levels = np.percentile(values, [0, 25, 50, 75, 100]).astype(float)

    levels = np.unique(levels)
    if levels.size < 2:
        base = float(levels[0]) if levels.size == 1 else 0.0
        eps = max(abs(base) * 1e-6, 1e-6)
        levels = np.array([base - eps, base + eps], dtype=float)
    return levels


def save_single_discrete_contour(df, var1, var2, target_col, colorbar_label,
                                 output_path, fixed_levels=None, label_fmt='%.4f',
                                 cbar_ticks=None, cbar_tick_fmt='%.2f'):
    """Create and save one contour map for a single variable pair."""
    grouped = df.groupby([var1, var2])[target_col].mean()
    pivot = grouped.reset_index().pivot(index=var2, columns=var1, values=target_col)

    x_vals = sorted(df[var1].dropna().unique())
    y_vals = sorted(df[var2].dropna().unique())

    z_vals = pivot.values.astype(float)
    values = z_vals.flatten()
    values = values[~np.isnan(values)]
    if values.size == 0:
        print(f"Skipped (no values): {output_path.name}")
        return False

    levels = build_levels(values, fixed_levels=fixed_levels)
    x_grid, y_grid = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

    # Fixed axis + colorbar layout keeps frame geometry consistent across files.
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    contourf = ax.contourf(
        x_grid, y_grid, z_vals,
        levels=levels, cmap='magma', alpha=0.8, extend='neither'
    )
    contour = ax.contour(
        x_grid, y_grid, z_vals,
        levels=levels, colors='white', linewidths=1.5, alpha=1.0
    )
    # Label only a subset of levels to keep readability with denser color bands.
    label_step = max(1, len(levels) // 5)
    label_levels = levels[::label_step]
    if levels[-1] not in label_levels:
        label_levels = np.append(label_levels, levels[-1])
    contour_labels = ax.clabel(contour, levels=label_levels, inline=True, fontsize=11, fmt=label_fmt)
    for txt in contour_labels:
        txt.set_fontweight('bold')

    x_edges = np.arange(len(x_vals) + 1) - 0.5
    y_edges = np.arange(len(y_vals) + 1) - 0.5
    for x_edge in x_edges:
        ax.axvline(x_edge, color='lightgray', linewidth=0.5, alpha=0.5)
    for y_edge in y_edges:
        ax.axhline(y_edge, color='lightgray', linewidth=0.5, alpha=0.5)

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
    ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

    cbar = fig.colorbar(contourf, cax=cax, ticks=cbar_ticks)
    if cbar_tick_fmt:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_tick_fmt))
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label=colorbar_label, cbar=cbar)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)
    ax.set_box_aspect(1)

    fig.tight_layout()
    save_figure_elsevier(output_path, fig=fig)
    plt.close(fig)
    print(f"Saved: {output_path.name}")
    return True


# ===================================================================
# Define variable pairs
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

# Build failure target once
df_full['is_failure'] = ((df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')).astype(int)
# Percent scale so fixed hydraulic levels [0, 25, 50, 75, 100] are valid.
df_full['is_failure_pct'] = df_full['is_failure'] * 100.0

# Standardized color scales per target across all variable-pair plots.
conv_min, conv_max = float(df_pass['CONV'].min()), float(df_pass['CONV'].max())
purity_min, purity_max = float(df_pass['PURITY'].min()), float(df_pass['PURITY'].max())
hydraulic_levels = np.linspace(0, 100, 11)
conv_levels = np.linspace(conv_min, conv_max, 13)
purity_levels = np.linspace(purity_min, purity_max, 13)
conv_ticks = np.linspace(conv_min, conv_max, 6)
purity_ticks = np.linspace(purity_min, purity_max, 6)

saved_count = 0

# ===================================================================
# HYDRAULIC FAILURES CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURES CONTOUR MAPS")
print("=" * 80)
for var1, var2 in hydraulic_pairs:
    out_path = output_dir / f"hydraulic_failures_{var1}_vs_{var2}_contour.png"
    saved = save_single_discrete_contour(
        df=df_full,
        var1=var1,
        var2=var2,
        target_col='is_failure_pct',
        colorbar_label='Failure Rate (%)',
        output_path=out_path,
        fixed_levels=hydraulic_levels,
        label_fmt='%.0f%%',
        cbar_ticks=[0, 25, 50, 75, 100],
        cbar_tick_fmt='%.0f'
    )
    saved_count += int(saved)

# ===================================================================
# CONVERSION CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("2. CONVERSION CONTOUR MAPS")
print("=" * 80)
for var1, var2 in conversion_pairs:
    out_path = output_dir / f"conversion_{var1}_vs_{var2}_contour.png"
    saved = save_single_discrete_contour(
        df=df_pass,
        var1=var1,
        var2=var2,
        target_col='CONV',
        colorbar_label='Conversion (%)',
        output_path=out_path,
        fixed_levels=conv_levels,
        label_fmt='%.2f',
        cbar_ticks=conv_ticks,
        cbar_tick_fmt='%.2f'
    )
    saved_count += int(saved)

# ===================================================================
# PURITY CONTOUR MAPS
# ===================================================================
print("\n" + "=" * 80)
print("3. PURITY CONTOUR MAPS")
print("=" * 80)
for var1, var2 in purity_pairs:
    out_path = output_dir / f"purity_{var1}_vs_{var2}_contour.png"
    saved = save_single_discrete_contour(
        df=df_pass,
        var1=var1,
        var2=var2,
        target_col='PURITY',
        colorbar_label='Purity (%)',
        output_path=out_path,
        fixed_levels=purity_levels,
        label_fmt='%.2f',
        cbar_ticks=purity_ticks,
        cbar_tick_fmt='%.2f'
    )
    saved_count += int(saved)

# Clean up temporary columns
df_full.drop(['is_failure', 'is_failure_pct'], axis=1, inplace=True)

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("DISCRETE CONTOUR MAP VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print(f"Generated {saved_count} individual contour images.")
print("\nVisualization details:")
print("  - One graph per output file")
print("  - Fixed plot frame geometry for consistent graph box size")
print("  - Contour lines based ONLY on discrete grid")
print("  - NO artificial smoothing or interpolation")
print("  - Arial font")
print("  - Standardized colorbar min/max per target (CONV and PURITY)")
print("  - 1000 DPI minimum (Elsevier export)")
print("=" * 80)
