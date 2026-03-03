import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde

from utils import create_binary_targets, filter_invalid_values, load_data

warnings.filterwarnings("ignore")

# Set up plotting style
sns.set_style("white")

# Global font settings
FONT_FAMILY = "Arial"
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 14

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

# Density legend spec: only show these values in legend
DENSITY_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
CMAP = plt.cm.magma

# Equilibrium marker
EQ_X = 98.1990652
EQ_Y = 99.461694


def apply_font(ax):
    ax.title.set_fontname(FONT_FAMILY)
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_fontname(FONT_FAMILY)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontname(FONT_FAMILY)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_SIZE)


def nice_round_limits(min_val, max_val, padding_ratio=0.05):
    span = max_val - min_val
    pad = span * padding_ratio
    nice_min = math.floor((min_val - pad) * 100) / 100
    nice_max = math.ceil((max_val + pad) * 100) / 100
    return nice_min, nice_max


def compute_scaled_density(x_vals, y_vals):
    """
    Compute 2D KDE and scale to [0, 0.5] so the requested legend
    values (0.1 ... 0.5) directly map to point colors.
    """
    points = np.vstack([x_vals, y_vals])

    # Fallback for degenerate covariance cases
    try:
        kde = gaussian_kde(points)
        density_raw = kde(points)
    except Exception:
        density_raw = np.ones_like(x_vals, dtype=float)

    if np.max(density_raw) <= 0:
        return np.zeros_like(density_raw, dtype=float)

    density_scaled = (density_raw / np.max(density_raw)) * 0.5
    return np.clip(density_scaled, 0.0, 0.5)


def make_density_legend_handles():
    handles = []
    for level in DENSITY_LEVELS:
        color = CMAP(level / 0.5)  # normalize 0..0.5 -> 0..1
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=9,
                label=f"{level:.1f}",
            )
        )
    return handles


def plot_density_scatter(df, title, output_stem, axis_limits, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["CONV_PCT"].to_numpy()
    y = df["PURITY_PCT"].to_numpy()
    density = compute_scaled_density(x, y)

    # Plot points sorted by density so dense points appear on top
    order = np.argsort(density)
    x_sorted = x[order]
    y_sorted = y[order]
    d_sorted = density[order]

    ax.scatter(
        x_sorted,
        y_sorted,
        c=d_sorted,
        cmap=CMAP,
        vmin=0.0,
        vmax=0.5,
        s=80,
        alpha=0.75,
        edgecolors="none",
    )

    # Equilibrium point (not included in legend by request)
    ax.scatter(
        [EQ_X],
        [EQ_Y],
        marker="*",
        s=220,
        color="#F81F8B",
        edgecolors="none",
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xlabel("Conversion (%)")
    ax.set_ylabel("Purity (%)")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlim(axis_limits[0], 100)
    ax.set_ylim(axis_limits[2], 100)
    apply_font(ax)

    # Legend only for density levels 0.1..0.5
    density_handles = make_density_legend_handles()
    ax.legend(
        handles=density_handles,
        title="Density",
        loc="best",
        frameon=True,
        prop={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE},
        title_fontproperties={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE},
    )

    plt.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.png", dpi=1000)
    fig.savefig(output_dir / f"{output_stem}.svg")
    plt.close(fig)


def main():
    print("\nLoading data...")
    df_full, df_pass = load_data(data_path="data/new_data.xlsx")
    df_full, df_pass = filter_invalid_values(df_full, df_pass)
    df_full = create_binary_targets(df_full)

    print(f"Full dataset: {len(df_full)} samples")
    print(f"Pass dataset: {len(df_pass)} samples")

    output_dir = Path("results/Scatter/MeAc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to percentage
    df_full["CONV_PCT"] = df_full["CONV"] * 100
    df_full["PURITY_PCT"] = df_full["PURITY"] * 100
    df_pass["CONV_PCT"] = df_pass["CONV"] * 100
    df_pass["PURITY_PCT"] = df_pass["PURITY"] * 100

    # Shared axis limits for direct visual comparison
    conv_all = np.concatenate([df_full["CONV_PCT"].dropna(), df_pass["CONV_PCT"].dropna()])
    purity_all = np.concatenate([df_full["PURITY_PCT"].dropna(), df_pass["PURITY_PCT"].dropna()])
    conv_min, conv_max = nice_round_limits(conv_all.min(), conv_all.max())
    pur_min, pur_max = nice_round_limits(purity_all.min(), purity_all.max())
    axis_limits = (conv_min, conv_max, pur_min, pur_max)

    plot_density_scatter(
        df=df_full,
        title="MeAc Conversion vs Purity (Full Dataset)",
        output_stem="full_dataset_density_scatter",
        axis_limits=axis_limits,
        output_dir=output_dir,
    )
    plot_density_scatter(
        df=df_pass,
        title="MeAc Conversion vs Purity (Pass-Only Dataset)",
        output_stem="pass_only_density_scatter",
        axis_limits=axis_limits,
        output_dir=output_dir,
    )

    print("\nDone.")
    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
