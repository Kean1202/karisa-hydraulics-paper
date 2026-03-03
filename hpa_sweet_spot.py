# -*- coding: utf-8 -*-
"""
HPA relation heatmaps (sweet-spot style, non-literal).

Question 1:
- Is there an HPA region that tends to PASS / WEEP / FLOOD?

Question 2:
- Is there an HPA region that tends to maximize CONVERSION and PURITY?

Outputs:
- results/hpa_sweet_spot/hpa_hydraulic_heatmap.png
- results/hpa_sweet_spot/hpa_quality_heatmap.png
- results/hpa_sweet_spot/hpa_combined_score_heatmap.png
- results/hpa_sweet_spot/hpa_bin_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import convert_to_percentage, filter_invalid_values, load_data, save_figure_elsevier


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=s.index)
    return (s - vmin) / (vmax - vmin)


def build_hpa_bins(df_full: pd.DataFrame, max_bins: int = 12):
    hpa = df_full["HPA"].dropna()
    n_bins = min(max_bins, int(hpa.nunique()))
    if n_bins < 3:
        raise RuntimeError("Not enough unique HPA values to build heatmap bins.")

    qbins = pd.qcut(hpa, q=n_bins, duplicates="drop")
    intervals = qbins.cat.categories
    edges = np.array([intervals[0].left] + [iv.right for iv in intervals], dtype=float)
    labels = [f"{iv.left:.3f}-{iv.right:.3f}" for iv in intervals]
    return edges, labels


def main():
    print("=" * 80)
    print("HPA SWEET-SPOT HEATMAPS")
    print("=" * 80)

    # Load new_data as requested by current project setup
    df_full, df_pass = load_data(data_path="data/new_data.xlsx")
    df_full, df_pass = filter_invalid_values(df_full, df_pass)
    df_pass = convert_to_percentage(df_pass, columns=["CONV", "PURITY"])

    if "HPA" not in df_full.columns or "HPA" not in df_pass.columns:
        raise RuntimeError("HPA column is required in both full_dataset and pass_only.")

    edges, bin_labels = build_hpa_bins(df_full, max_bins=12)

    df_full = df_full.copy()
    df_pass = df_pass.copy()
    df_full["HPA_BIN"] = pd.cut(df_full["HPA"], bins=edges, include_lowest=True, labels=bin_labels)
    df_pass["HPA_BIN"] = pd.cut(df_pass["HPA"], bins=edges, include_lowest=True, labels=bin_labels)

    # -----------------------------------------------------------------
    # 1) Hydraulic tendency heatmap: PASS/WEEP/FLOOD rates per HPA bin
    # -----------------------------------------------------------------
    counts = (
        df_full.groupby(["DESC", "HPA_BIN"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["PASS", "WEEP", "FLOOD"], fill_value=0)
    )
    col_totals = counts.sum(axis=0).replace(0, np.nan)
    hydraulic_pct = counts.divide(col_totals, axis=1) * 100.0

    # -----------------------------------------------------------------
    # 2) Quality tendency heatmap: mean CONV/PURITY per HPA bin
    # -----------------------------------------------------------------
    quality_agg = (
        df_pass.groupby("HPA_BIN", observed=False)
        .agg(
            conversion_pct=("CONV", "mean"),
            purity_pct=("PURITY", "mean"),
            n_pass=("CONV", "size"),
        )
    )

    quality_matrix = pd.DataFrame(
        index=["Conversion (%)", "Purity (%)"],
        columns=bin_labels,
        dtype=float,
    )
    quality_matrix.loc["Conversion (%)", :] = quality_agg["conversion_pct"].reindex(bin_labels).values
    quality_matrix.loc["Purity (%)", :] = quality_agg["purity_pct"].reindex(bin_labels).values

    # -----------------------------------------------------------------
    # 3) Combined score heatmap strip (PASS high, WEEP/FLOOD low, CONV/PURITY high)
    # -----------------------------------------------------------------
    pass_rate = hydraulic_pct.loc["PASS", :].reindex(bin_labels)
    weep_rate = hydraulic_pct.loc["WEEP", :].reindex(bin_labels)
    flood_rate = hydraulic_pct.loc["FLOOD", :].reindex(bin_labels)
    conv = quality_matrix.loc["Conversion (%)", :].reindex(bin_labels)
    purity = quality_matrix.loc["Purity (%)", :].reindex(bin_labels)

    score_df = pd.DataFrame(
        {
            "pass_norm": minmax(pass_rate),
            "weep_norm": minmax(-weep_rate),
            "flood_norm": minmax(-flood_rate),
            "conv_norm": minmax(conv),
            "purity_norm": minmax(purity),
        }
    )
    combined_score = score_df.mean(axis=1, skipna=False)
    combined_matrix = pd.DataFrame([combined_score.values], index=["Combined score (0-1)"], columns=bin_labels)

    # -----------------------------------------------------------------
    # Save summary table for exact values
    # -----------------------------------------------------------------
    summary = pd.DataFrame(index=bin_labels)
    summary.index.name = "hpa_bin"
    summary["pass_rate_pct"] = pass_rate
    summary["weep_rate_pct"] = weep_rate
    summary["flood_rate_pct"] = flood_rate
    summary["conversion_pct"] = conv
    summary["purity_pct"] = purity
    summary["combined_score"] = combined_score
    summary["n_full"] = col_totals.reindex(bin_labels)
    summary["n_pass"] = quality_agg["n_pass"].reindex(bin_labels)

    out_dir = Path("results/hpa_sweet_spot")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(out_dir / "hpa_bin_summary.csv", index=False)

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Arial"

    # Hydraulic heatmap
    fig1, ax1 = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        hydraulic_pct,
        ax=ax1,
        cmap="magma",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Rate (%)"},
    )
    ax1.set_xlabel("HPA bin")
    ax1.set_ylabel("Hydraulic outcome")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    save_figure_elsevier(out_dir / "hpa_hydraulic_heatmap.png", fig=fig1)
    plt.close(fig1)

    # Quality heatmap
    fig2, ax2 = plt.subplots(figsize=(16, 3.5))
    qmin = np.nanmin(quality_matrix.values)
    qmax = np.nanmax(quality_matrix.values)
    sns.heatmap(
        quality_matrix,
        ax=ax2,
        cmap="magma",
        vmin=qmin,
        vmax=qmax,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Value (%)"},
    )
    ax2.set_xlabel("HPA bin")
    ax2.set_ylabel("Quality metric")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    save_figure_elsevier(out_dir / "hpa_quality_heatmap.png", fig=fig2)
    plt.close(fig2)

    # Combined score strip
    fig3, ax3 = plt.subplots(figsize=(16, 2.8))
    sns.heatmap(
        combined_matrix,
        ax=ax3,
        cmap="magma",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Combined score"},
    )
    ax3.set_xlabel("HPA bin")
    ax3.set_ylabel("")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    save_figure_elsevier(out_dir / "hpa_combined_score_heatmap.png", fig=fig3)
    plt.close(fig3)

    if combined_score.notna().any():
        best_bin = combined_score.idxmax()
        print(f"Best combined HPA bin: {best_bin} (score={combined_score.loc[best_bin]:.3f})")
    else:
        print("No combined-score bins available (missing overlap across metrics).")

    print(f"Saved: {(out_dir / 'hpa_hydraulic_heatmap.png').as_posix()}")
    print(f"Saved: {(out_dir / 'hpa_quality_heatmap.png').as_posix()}")
    print(f"Saved: {(out_dir / 'hpa_combined_score_heatmap.png').as_posix()}")
    print(f"Saved: {(out_dir / 'hpa_bin_summary.csv').as_posix()}")


if __name__ == "__main__":
    main()
