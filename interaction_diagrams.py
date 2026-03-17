# -*- coding: utf-8 -*-
"""
Interaction Diagrams - Variable Pair Interactions for All Targets

Creates one image per variable pair (individual plots) for:
- HYDRAULIC FAILURES (WEEP + FLOOD): colored by DESC
- CONVERSION: colored by CONV
- PURITY: colored by PURITY
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    DESC_COLORS,
    convert_to_percentage,
    create_binary_targets,
    filter_invalid_values,
    format_axis_for_paper,
    load_data,
    save_figure_elsevier,
)

warnings.filterwarnings("ignore")

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Arial"

CONTINUOUS_CMAP = "magma"
INTERACTION_MARKER_SIZE = 70
INTERACTION_LEGEND_FONTSIZE = 14
INTERACTION_LABEL_FONTSIZE = 20
INTERACTION_TICK_FONTSIZE = 16

# Per-variable decimal precision for tick labels (None = auto)
_VAR_DECIMALS = {
    'HDIAM': 6,
    'WEIRHT': 4,
}

print("=" * 80)
print("INTERACTION DIAGRAMS - INDIVIDUAL PLOTS")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full = create_binary_targets(df_full)
df_pass = convert_to_percentage(df_pass, columns=["CONV", "PURITY"])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples (CONV & PURITY converted to %)")

# Output directories
base_output_dir = Path("results/interaction_diagrams")
hydraulic_dir = base_output_dir / "hydraulic"
conversion_dir = base_output_dir / "conversion"
purity_dir = base_output_dir / "purity"

for d in [hydraulic_dir, conversion_dir, purity_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Variable pairs
hydraulic_pairs = [
    ("NHOLES", "HDIAM"),
    ("DIAM", "HDIAM"),
    ("TRAYSPC", "HDIAM"),
    ("WEIRHT", "HDIAM"),
    ("NHOLES", "DIAM"),
    ("NHOLES", "TRAYSPC"),
]

conversion_pairs = [
    ("DIAM", "WEIRHT"),
    ("DIAM", "NPASS"),
    ("DIAM", "TRAYSPC"),
    ("WEIRHT", "NPASS"),
    ("WEIRHT", "TRAYSPC"),
    ("NPASS", "TRAYSPC"),
]

purity_pairs = [
    ("DIAM", "WEIRHT"),
    ("DIAM", "NPASS"),
    ("DIAM", "TRAYSPC"),
    ("WEIRHT", "NPASS"),
    ("WEIRHT", "TRAYSPC"),
    ("NPASS", "TRAYSPC"),
]


def _index_map(df, col):
    """Return sorted unique values and a value->index dict for a column."""
    vals = sorted(df[col].dropna().unique())
    return vals, {v: i for i, v in enumerate(vals)}


def _tick_labels(vals, decimals=None):
    if decimals is not None:
        return [f'{v:.{decimals}f}' for v in vals]
    return [f'{int(v)}' if v == int(v) else f'{v:g}' for v in vals]


def save_hydraulic_plot(var1, var2):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Map raw values to integer indices so each grid cell is equally sized,
    # matching the discrete contour map layout for clean overlay.
    x_vals, x_map = _index_map(df_full, var1)
    y_vals, y_map = _index_map(df_full, var2)

    df_plot = df_full.copy()
    df_plot['_x'] = df_plot[var1].map(x_map)
    df_plot['_y'] = df_plot[var2].map(y_map)

    sns.scatterplot(
        data=df_plot,
        x='_x',
        y='_y',
        hue="DESC",
        palette=DESC_COLORS,
        alpha=0.5,
        s=INTERACTION_MARKER_SIZE,
        ax=ax,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(_tick_labels(x_vals, decimals=_VAR_DECIMALS.get(var1)), rotation=45, ha='right')
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(_tick_labels(y_vals, decimals=_VAR_DECIMALS.get(var2)))
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)

    format_axis_for_paper(ax, xlabel=var1, ylabel=var2,
                          label_fontsize=INTERACTION_LABEL_FONTSIZE,
                          tick_fontsize=INTERACTION_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect(1)
    ax.legend(loc="best", fontsize=INTERACTION_LEGEND_FONTSIZE)

    output_path = hydraulic_dir / f"hydraulic_{var1}_vs_{var2}.png"
    save_figure_elsevier(output_path, fig=fig)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def save_continuous_plot(var1, var2, target_col, label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Map raw values to integer indices so each grid cell is equally sized,
    # matching the discrete contour map layout for clean overlay.
    x_vals, x_map = _index_map(df_pass, var1)
    y_vals, y_map = _index_map(df_pass, var2)

    x_idx = df_pass[var1].map(x_map)
    y_idx = df_pass[var2].map(y_map)

    scatter = ax.scatter(
        x_idx,
        y_idx,
        c=df_pass[target_col],
        cmap=CONTINUOUS_CMAP,
        alpha=0.5,
        s=INTERACTION_MARKER_SIZE,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(_tick_labels(x_vals, decimals=_VAR_DECIMALS.get(var1)), rotation=45, ha='right')
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(_tick_labels(y_vals, decimals=_VAR_DECIMALS.get(var2)))
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label=label, cbar=cbar,
                          label_fontsize=INTERACTION_LABEL_FONTSIZE,
                          tick_fontsize=INTERACTION_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect(1)

    output_path = output_dir / f"{target_col.lower()}_{var1}_vs_{var2}.png"
    save_figure_elsevier(output_path, fig=fig)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURE INTERACTIONS")
print("=" * 80)
for var1, var2 in hydraulic_pairs:
    save_hydraulic_plot(var1, var2)

print("\n" + "=" * 80)
print("2. CONVERSION INTERACTIONS")
print("=" * 80)
for var1, var2 in conversion_pairs:
    save_continuous_plot(var1, var2, "CONV", "Conversion (%)", conversion_dir)

print("\n" + "=" * 80)
print("3. PURITY INTERACTIONS")
print("=" * 80)
for var1, var2 in purity_pairs:
    save_continuous_plot(var1, var2, "PURITY", "Purity (%)", purity_dir)

print("\n" + "=" * 80)
print("INTERACTION DIAGRAMS COMPLETE!")
print("=" * 80)
print(f"\nSaved outputs in: {base_output_dir.absolute()}")
print(f"- Hydraulic:  {len(hydraulic_pairs)} files")
print(f"- Conversion: {len(conversion_pairs)} files")
print(f"- Purity:     {len(purity_pairs)} files")
