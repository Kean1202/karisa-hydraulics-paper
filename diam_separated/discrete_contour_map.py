# -*- coding: utf-8 -*-
"""
Discrete Contour Maps - DIAM-Separated Version

Same as root discrete_contour_map.py but:
- Dataset filtered to a single DIAM value (KARISA_DIAM_FILTER env var)
- Pairs involving DIAM are excluded (DIAM is constant within the slice)
- Outputs go to results/diam_separated/DIAM_{value}/discrete_contour_maps/

Run via main_diam_separated.py (do not run directly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_data, filter_invalid_values, filter_by_diam,
    create_binary_targets, format_axis_for_paper,
    convert_to_percentage, save_figure_elsevier,
    DIAM_FILTER_STR,
)

sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'

diam_val = float(DIAM_FILTER_STR)
DIAM_LABEL = f"DIAM_{diam_val:g}"
DIAM_OUT = Path(f"results/diam_separated/{DIAM_LABEL}")

print("=" * 80)
print(f"DISCRETE CONTOUR MAP VISUALIZATIONS - DIAM-Separated  [{DIAM_LABEL}]")
print("=" * 80)

print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/new_data.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = filter_by_diam(df_full, df_pass)
df_full = create_binary_targets(df_full)
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

print(f"Full dataset: {len(df_full)} samples")
print(f"Pass dataset: {len(df_pass)} samples")

output_dir = DIAM_OUT / "discrete_contour_maps"
output_dir.mkdir(parents=True, exist_ok=True)


def build_levels(values, fixed_levels=None):
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


CONTOUR_LABEL_FONTSIZE = 20
CONTOUR_TICK_FONTSIZE = 16

# Per-variable decimal precision for tick labels (None = auto)
_VAR_DECIMALS = {
    'HDIAM': 6,
    'WEIRHT': 4,
}


def _tick_label(v, decimals=None):
    if decimals is not None:
        return f'{v:.{decimals}f}'
    return f'{int(v)}' if v == int(v) else f'{v:g}'


def save_single_discrete_contour(df, var1, var2, target_col, colorbar_label,
                                  output_path, fixed_levels=None, label_fmt='%.4f',
                                  cbar_ticks=None, cbar_tick_fmt='%.2f'):
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
    # White contour lines at 25%, 50%, 75% of the value range (evenly spaced).
    lo, hi = levels[0], levels[-1]
    line_levels = np.array([lo + 0.25*(hi-lo), lo + 0.50*(hi-lo), lo + 0.75*(hi-lo)])
    x_grid, y_grid = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    contourf = ax.contourf(x_grid, y_grid, z_vals, levels=levels, cmap='magma_r', alpha=0.8, extend='neither')
    contour = ax.contour(x_grid, y_grid, z_vals, levels=line_levels, colors='white', linewidths=1.5, alpha=1.0)
    contour_labels = ax.clabel(contour, levels=line_levels, inline=True, fontsize=14, fmt=label_fmt)
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
    ax.set_xticklabels([_tick_label(v, _VAR_DECIMALS.get(var1)) for v in x_vals], rotation=45, ha='right')
    ax.set_yticklabels([_tick_label(v, _VAR_DECIMALS.get(var2)) for v in y_vals])

    cbar = fig.colorbar(contourf, cax=cax, ticks=cbar_ticks)
    if cbar_tick_fmt:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_tick_fmt))
    format_axis_for_paper(ax, xlabel=var1, ylabel=var2, colorbar_label=colorbar_label, cbar=cbar,
                          label_fontsize=CONTOUR_LABEL_FONTSIZE,
                          tick_fontsize=CONTOUR_TICK_FONTSIZE)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)
    ax.set_box_aspect(1)

    fig.tight_layout()
    save_figure_elsevier(output_path, fig=fig)
    plt.close(fig)
    print(f"Saved: {output_path.name}")
    return True


# DIAM is constant within this slice — exclude pairs involving DIAM
hydraulic_pairs = [
    ('NHOLES', 'HDIAM'),
    ('TRAYSPC', 'HDIAM'),
    ('WEIRHT', 'HDIAM'),
    ('NHOLES', 'TRAYSPC'),
]

conversion_pairs = [
    ('WEIRHT', 'NPASS'),
    ('WEIRHT', 'TRAYSPC'),
    ('NPASS', 'TRAYSPC'),
]

purity_pairs = [
    ('WEIRHT', 'NPASS'),
    ('WEIRHT', 'TRAYSPC'),
    ('NPASS', 'TRAYSPC'),
]

df_full['is_failure'] = ((df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')).astype(int)
df_full['is_failure_pct'] = df_full['is_failure'] * 100.0

conv_min, conv_max = float(df_pass['CONV'].min()), float(df_pass['CONV'].max())
purity_min, purity_max = float(df_pass['PURITY'].min()), float(df_pass['PURITY'].max())
hydraulic_levels = np.linspace(0, 100, 11)
conv_levels = np.linspace(conv_min, conv_max, 13)
purity_levels = np.linspace(purity_min, purity_max, 13)
conv_ticks = np.linspace(conv_min, conv_max, 6)
purity_ticks = np.linspace(purity_min, purity_max, 6)

saved_count = 0

print("\n" + "=" * 80)
print("1. HYDRAULIC FAILURES CONTOUR MAPS")
print("=" * 80)
for var1, var2 in hydraulic_pairs:
    out_path = output_dir / f"hydraulic_failures_{var1}_vs_{var2}_contour.png"
    saved_count += int(save_single_discrete_contour(
        df=df_full, var1=var1, var2=var2,
        target_col='is_failure_pct', colorbar_label='Failure Rate (%)',
        output_path=out_path, fixed_levels=hydraulic_levels,
        label_fmt='%.0f%%', cbar_ticks=[0, 25, 50, 75, 100], cbar_tick_fmt='%.0f'
    ))

print("\n" + "=" * 80)
print("2. CONVERSION CONTOUR MAPS")
print("=" * 80)
for var1, var2 in conversion_pairs:
    out_path = output_dir / f"conversion_{var1}_vs_{var2}_contour.png"
    saved_count += int(save_single_discrete_contour(
        df=df_pass, var1=var1, var2=var2,
        target_col='CONV', colorbar_label='Conversion (%)',
        output_path=out_path, fixed_levels=conv_levels,
        label_fmt='%.2f', cbar_ticks=conv_ticks, cbar_tick_fmt='%.2f'
    ))

print("\n" + "=" * 80)
print("3. PURITY CONTOUR MAPS")
print("=" * 80)
for var1, var2 in purity_pairs:
    out_path = output_dir / f"purity_{var1}_vs_{var2}_contour.png"
    saved_count += int(save_single_discrete_contour(
        df=df_pass, var1=var1, var2=var2,
        target_col='PURITY', colorbar_label='Purity (%)',
        output_path=out_path, fixed_levels=purity_levels,
        label_fmt='%.2f', cbar_ticks=purity_ticks, cbar_tick_fmt='%.2f'
    ))

df_full.drop(['is_failure', 'is_failure_pct'], axis=1, inplace=True)

print("\n" + "=" * 80)
print(f"DISCRETE CONTOUR MAPS COMPLETE  [{DIAM_LABEL}]")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print(f"Generated {saved_count} individual contour images.")
