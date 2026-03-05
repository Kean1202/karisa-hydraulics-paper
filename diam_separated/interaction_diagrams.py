# -*- coding: utf-8 -*-
"""
Interaction Diagrams - DIAM-Separated Version

Same as root interaction_diagrams.py but:
- Dataset filtered to a single DIAM value (KARISA_DIAM_FILTER env var)
- Pairs involving DIAM are excluded (DIAM is constant within the slice)
- Outputs go to results/diam_separated/DIAM_{value}/interaction_diagrams/

Run via main_diam_separated.py (do not run directly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    DESC_COLORS,
    DIAM_FILTER_STR,
    convert_to_percentage,
    create_binary_targets,
    filter_invalid_values,
    filter_by_diam,
    format_axis_for_paper,
    load_data,
    save_figure_elsevier,
)

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Arial"

CONTINUOUS_CMAP = "magma"

diam_val = float(DIAM_FILTER_STR)
DIAM_LABEL = f"DIAM_{diam_val:g}"
DIAM_OUT = Path(f"results/diam_separated/{DIAM_LABEL}")

print("=" * 80)
print(f"INTERACTION DIAGRAMS - DIAM-Separated  [{DIAM_LABEL}]")
print("=" * 80)

print("\nLoading data...")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = filter_by_diam(df_full, df_pass)
df_full = create_binary_targets(df_full)
df_pass = convert_to_percentage(df_pass, columns=["CONV", "PURITY"])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples")

# Output directories
base_output_dir = DIAM_OUT / "interaction_diagrams"
hydraulic_dir = base_output_dir / "hydraulic"
conversion_dir = base_output_dir / "conversion"
purity_dir = base_output_dir / "purity"

for d in [hydraulic_dir, conversion_dir, purity_dir]:
    d.mkdir(parents=True, exist_ok=True)

# DIAM is constant within this slice — exclude pairs that involve DIAM
hydraulic_pairs = [
    ("NHOLES", "HDIAM"),
    ("TRAYSPC", "HDIAM"),
    ("WEIRHT", "HDIAM"),
    ("NHOLES", "TRAYSPC"),
]

conversion_pairs = [
    ("WEIRHT", "NPASS"),
    ("WEIRHT", "TRAYSPC"),
    ("NPASS", "TRAYSPC"),
]

purity_pairs = [
    ("WEIRHT", "NPASS"),
    ("WEIRHT", "TRAYSPC"),
    ("NPASS", "TRAYSPC"),
]


def _index_map(df, col):
    """Return sorted unique values and a value->index dict for a column."""
    vals = sorted(df[col].dropna().unique())
    return vals, {v: i for i, v in enumerate(vals)}


def _tick_labels(vals):
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
        data=df_plot, x='_x', y='_y',
        hue="DESC", palette=DESC_COLORS,
        alpha=0.5, s=30, ax=ax,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(_tick_labels(x_vals), rotation=45, ha='right')
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(_tick_labels(y_vals))
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)

    format_axis_for_paper(ax, xlabel=var1, ylabel=var2)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect(1)
    ax.legend(loc="best", fontsize=9)
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
        x_idx, y_idx,
        c=df_pass[target_col], cmap=CONTINUOUS_CMAP,
        alpha=0.5, s=30,
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(_tick_labels(x_vals), rotation=45, ha='right')
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(_tick_labels(y_vals))
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label=label, cbar=cbar)
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
print(f"INTERACTION DIAGRAMS COMPLETE  [{DIAM_LABEL}]")
print("=" * 80)
print(f"\nSaved outputs in: {base_output_dir.absolute()}")
print(f"- Hydraulic:  {len(hydraulic_pairs)} files")
print(f"- Conversion: {len(conversion_pairs)} files")
print(f"- Purity:     {len(purity_pairs)} files")
