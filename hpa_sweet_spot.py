# -*- coding: utf-8 -*-
"""
HPA sweet-spot heatmaps.

Question 1: Is there an HPA region that tends to PASS / WEEP / FLOOD?
Question 2: Is there an HPA region that tends to maximize CONV and PURITY?

Outputs:
  results/hpa_sweet_spot/hpa_hydraulic_heatmap.png
  results/hpa_sweet_spot/hpa_quality_heatmap.png
  results/hpa_sweet_spot/hpa_combined_score_bar.png
  results/hpa_sweet_spot/hpa_bin_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import convert_to_percentage, filter_invalid_values, load_data, save_figure_elsevier

# Fixed, physically meaningful HPA bins (after HPA_MIN=0.01 filter)
# HPA = fraction of tray area covered by holes, range 0.01–0.90
HPA_EDGES  = [0.01, 0.05, 0.10, 0.20, 0.35, 0.55, 0.90]
HPA_LABELS = ["0.01–0.05", "0.05–0.10", "0.10–0.20", "0.20–0.35", "0.35–0.55", "0.55–0.90"]

FONT = "Arial"
sns.set_style("whitegrid")
plt.rcParams["font.family"] = FONT


def minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1], returns NaN series if flat or all-NaN."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def main():
    print("=" * 80)
    print("HPA SWEET-SPOT HEATMAPS")
    print("=" * 80)

    df_full, df_pass = load_data()
    df_full, df_pass = filter_invalid_values(df_full, df_pass)
    df_pass = convert_to_percentage(df_pass, columns=["CONV", "PURITY"])

    if "HPA" not in df_full.columns or "HPA" not in df_pass.columns:
        raise RuntimeError("HPA column is required in both full_dataset and pass_only.")

    df_full = df_full.copy()
    df_pass = df_pass.copy()
    df_full["HPA_BIN"] = pd.cut(
        df_full["HPA"], bins=HPA_EDGES, include_lowest=True, labels=HPA_LABELS
    )
    df_pass["HPA_BIN"] = pd.cut(
        df_pass["HPA"], bins=HPA_EDGES, include_lowest=True, labels=HPA_LABELS
    )

    # ------------------------------------------------------------------
    # 1) Hydraulic rates per HPA bin
    # ------------------------------------------------------------------
    counts = (
        df_full.groupby(["DESC", "HPA_BIN"], observed=False)
        .size()
        .unstack(fill_value=0)
        # FIX: FLOOD→WEEP→PASS order so magma reads dark=danger, bright=safe
        .reindex(index=["FLOOD", "WEEP", "PASS"], fill_value=0)
    )
    col_totals = counts.sum(axis=0).replace(0, np.nan)
    hydraulic_pct = counts.divide(col_totals, axis=1) * 100.0

    # ------------------------------------------------------------------
    # 2) Mean CONV and PURITY per HPA bin (pass-only data)
    # ------------------------------------------------------------------
    quality_agg = (
        df_pass.groupby("HPA_BIN", observed=False)
        .agg(
            conversion_pct=("CONV",   "mean"),
            purity_pct    =("PURITY", "mean"),
            n_pass        =("CONV",   "size"),
        )
    )
    conv_row   = quality_agg["conversion_pct"].reindex(HPA_LABELS)
    purity_row = quality_agg["purity_pct"].reindex(HPA_LABELS)

    # ------------------------------------------------------------------
    # 3) Combined score
    # FIX: skipna=True so bins with only hydraulic data (no PASS quality
    #      data) still receive a partial score instead of going fully NaN
    # ------------------------------------------------------------------
    pass_rate  = hydraulic_pct.loc["PASS",  :].reindex(HPA_LABELS)
    weep_rate  = hydraulic_pct.loc["WEEP",  :].reindex(HPA_LABELS)
    flood_rate = hydraulic_pct.loc["FLOOD", :].reindex(HPA_LABELS)

    score_df = pd.DataFrame({
        "pass_norm":   minmax(pass_rate),
        "weep_norm":   minmax(-weep_rate),
        "flood_norm":  minmax(-flood_rate),
        "conv_norm":   minmax(conv_row),
        "purity_norm": minmax(purity_row),
    })
    combined_score = score_df.mean(axis=1, skipna=True)

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    out_dir = Path("results/hpa_sweet_spot")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(index=HPA_LABELS)
    summary.index.name  = "hpa_bin"
    summary["pass_rate_pct"]  = pass_rate
    summary["weep_rate_pct"]  = weep_rate
    summary["flood_rate_pct"] = flood_rate
    summary["conversion_pct"] = conv_row
    summary["purity_pct"]     = purity_row
    summary["combined_score"] = combined_score
    summary["n_full"] = col_totals.reindex(HPA_LABELS)
    summary["n_pass"] = quality_agg["n_pass"].reindex(HPA_LABELS)
    summary.reset_index().to_csv(out_dir / "hpa_bin_summary.csv", index=False)

    # ==================================================================
    # PLOT 1 — Hydraulic heatmap
    # ==================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        hydraulic_pct,
        ax=ax1,
        cmap="magma",
        vmin=0, vmax=100,
        annot=True, fmt=".1f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Rate (%)"},
    )
    ax1.set_xlabel("HPA bin", fontsize=14, fontfamily=FONT)
    ax1.set_ylabel("Hydraulic outcome", fontsize=14, fontfamily=FONT)
    ax1.tick_params(labelsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    save_figure_elsevier(out_dir / "hpa_hydraulic_heatmap.png", fig=fig1)
    plt.close(fig1)

    # ==================================================================
    # PLOT 2 — Quality heatmap, separate colour scale per metric
    # FIX: the old single vmin/vmax across both rows made PURITY (which
    #      has smaller variance) appear nearly flat.  Two stacked subplots
    #      with independent scales show real variation in each row.
    # ==================================================================
    fig2, (ax_conv, ax_pur) = plt.subplots(
        2, 1, figsize=(12, 5), gridspec_kw={"hspace": 0.6}
    )
    conv_mat   = pd.DataFrame([conv_row.values],   index=["Conversion (%)"], columns=HPA_LABELS)
    purity_mat = pd.DataFrame([purity_row.values], index=["Purity (%)"],     columns=HPA_LABELS)

    for ax, mat, label, row in [
        (ax_conv, conv_mat,   "Conversion (%)", conv_row),
        (ax_pur,  purity_mat, "Purity (%)",     purity_row),
    ]:
        sns.heatmap(
            mat,
            ax=ax,
            cmap="magma",
            vmin=row.min(), vmax=row.max(),
            annot=True, fmt=".2f",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": label},
        )
        ax.set_xlabel("" if ax is ax_conv else "HPA bin", fontsize=14, fontfamily=FONT)
        ax.set_ylabel("", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.tick_params(labelsize=12)

    save_figure_elsevier(out_dir / "hpa_quality_heatmap.png", fig=fig2)
    plt.close(fig2)

    # ==================================================================
    # PLOT 3 — Combined score as a horizontal bar chart
    # FIX: a single-row heatmap strip added no visual value over the
    #      annotations; a bar chart makes the ranking immediately obvious.
    # ==================================================================
    valid = combined_score.notna()
    score_vals = combined_score.values.astype(float)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    norm_vals = np.where(valid, score_vals / np.nanmax(score_vals), 0.0)
    colors = plt.cm.magma(norm_vals)
    bars = ax3.barh(
        HPA_LABELS, score_vals,
        color=colors, edgecolor="white", linewidth=0.5
    )
    ax3.set_xlabel("Combined score (0–1)", fontsize=14, fontfamily=FONT)
    ax3.set_ylabel("HPA bin", fontsize=14, fontfamily=FONT)
    ax3.set_xlim(0, 1.08)
    ax3.tick_params(labelsize=12)
    ax3.invert_yaxis()

    for bar, val, ok in zip(bars, score_vals, valid):
        if ok:
            ax3.text(
                val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=11, fontfamily=FONT,
            )

    plt.tight_layout()
    save_figure_elsevier(out_dir / "hpa_combined_score_bar.png", fig=fig3)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    if combined_score.notna().any():
        best = combined_score.idxmax()
        print(f"\nBest HPA bin:  {best}  (score = {combined_score[best]:.3f})")
    else:
        print("\nNo combined-score data available.")

    print(f"Saved to: {out_dir.as_posix()}/")


if __name__ == "__main__":
    main()
