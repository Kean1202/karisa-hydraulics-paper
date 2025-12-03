# -*- coding: utf-8 -*-
"""
Hydraulic Gradient Visualizations - 3 Different Approaches

Creates 3 different gradient-style visualizations for hydraulic failures to make
WEEP/PASS/FLOOD outcomes look more "gradient" instead of discrete scatter plots.

OPTION 1: Probability Gradient Background
- Pure smooth 0-100% failure rate gradient
- Contour lines at 25%, 50%, 75%
- Most gradient-like appearance

OPTION 2: Three-Class Probability Heatmap
- Dominant outcome (PASS/WEEP/FLOOD) with certainty intensity
- Faded colors = low certainty/mixed outcomes
- Shows uncertainty visually

OPTION 3: Hybrid Gradient + Scatter + Boundaries
- Layer 1: Failure probability gradient background
- Layer 2: Semi-transparent scatter points
- Layer 3: XGBoost decision boundaries
- Combines all approaches

Generates 3 separate image files, each with 6 subplots showing the critical
hydraulic variable pairs from interaction_diagrams.py.

Made with love for Karisa - whatever the professor wants!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import load_data, filter_invalid_values, deduplicate_data

# Import XGBoost for decision boundaries (Option 3)
from xgboost import XGBClassifier

# Set up plotting style
sns.set_style("white")

# Import magma colormap
import matplotlib.cm as cm
magma_cmap = cm.get_cmap('magma')
DESC_COLORS = {
    'PASS': magma_cmap(0.2),
    'WEEP': magma_cmap(0.5),
    'FLOOD': magma_cmap(0.9)
}

print("=" * 80)
print("HYDRAULIC GRADIENT VISUALIZATIONS - 3 APPROACHES")
print("Generating 3 images, each with 6 gradient-style plots")
print("=" * 80)

# Load and prepare data
print("\nLoading data...")
df_full, df_pass = load_data(data_path="data/karisa_paper.xlsx")
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)

print(f"Full dataset: {len(df_full)} samples (after deduplication)")

# Create output directory
output_dir = Path("results/whatever_professor_wants")
output_dir.mkdir(parents=True, exist_ok=True)

# Define variable pairs (same as interaction_diagrams.py hydraulic section)
hydraulic_pairs = [
    ('NHOLES', 'HDIAM'),
    ('DIAM', 'HDIAM'),
    ('TRAYSPC', 'HDIAM'),
    ('WEIRHT', 'HDIAM'),
    ('NHOLES', 'DIAM'),
    ('NHOLES', 'TRAYSPC')
]

# ===================================================================
# OPTION 1: Probability Gradient Background
# Pure smooth 0-100% failure rate gradient
# ===================================================================
print("\n" + "=" * 80)
print("OPTION 1: Probability Gradient Background")
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

        # Define smooth gradient levels (0-100%)
        levels = np.linspace(0, 100, 21)  # 21 levels for smooth gradient

        # Filled contour plot - smooth gradient
        contourf = ax.contourf(X, Y, pivot.values, levels=levels,
                               cmap='magma', alpha=0.9, extend='neither')

        # Add contour lines at key thresholds
        contour = ax.contour(X, Y, pivot.values, levels=[25, 50, 75],
                            colors='white', linewidths=1.5, alpha=0.7)
        ax.clabel(contour, inline=True, fontsize=9, fmt='%.0f%%')

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nFailure Probability', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Failure Rate (%)', fontsize=10)

# Clean up temporary column
df_full.drop('is_failure', axis=1, inplace=True)

fig.suptitle('Option 1: Probability Gradient Background\n(Smooth 0-100% failure risk gradient)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
output_path_1 = output_dir / 'option1_probability_gradient.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_1}")
plt.close()

# ===================================================================
# OPTION 2: Three-Class Probability Heatmap
# Dominant outcome with certainty intensity
# ===================================================================
print("\n" + "=" * 80)
print("OPTION 2: Three-Class Probability Heatmap")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    if var1 in df_full.columns and var2 in df_full.columns:
        ax = axes[idx]

        # Calculate probability of each outcome for each combination
        results = []
        for (v1, v2), group in df_full.groupby([var1, var2]):
            desc_values = group['DESC']
            total = len(desc_values)
            pass_pct = (desc_values == 'PASS').sum() / total
            weep_pct = (desc_values == 'WEEP').sum() / total
            flood_pct = (desc_values == 'FLOOD').sum() / total

            # Determine dominant class
            probs = {'PASS': pass_pct, 'WEEP': weep_pct, 'FLOOD': flood_pct}
            dominant = max(probs, key=probs.get)
            max_prob = probs[dominant]

            results.append({
                var1: v1,
                var2: v2,
                'dominant': dominant,
                'certainty': max_prob,
                'pass_pct': pass_pct,
                'weep_pct': weep_pct,
                'flood_pct': flood_pct
            })

        grouped = pd.DataFrame(results)

        # Get actual discrete values
        x_vals = sorted(df_full[var1].unique())
        y_vals = sorted(df_full[var2].unique())

        # Create RGB image array
        img = np.zeros((len(y_vals), len(x_vals), 3))
        certainty_grid = np.zeros((len(y_vals), len(x_vals)))

        for _, row in grouped.iterrows():
            x_idx = x_vals.index(row[var1])
            y_idx = y_vals.index(row[var2])

            # Map outcome to magma color
            if row['dominant'] == 'PASS':
                base_color = magma_cmap(0.2)[:3]
            elif row['dominant'] == 'WEEP':
                base_color = magma_cmap(0.5)[:3]
            else:  # FLOOD
                base_color = magma_cmap(0.9)[:3]

            # Scale by certainty (higher certainty = more vivid)
            certainty = row['certainty']
            # Mix with gray for low certainty
            color = np.array(base_color) * certainty + np.array([0.5, 0.5, 0.5]) * (1 - certainty)

            img[y_idx, x_idx] = color
            certainty_grid[y_idx, x_idx] = certainty

        # Display as image
        extent = [-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5]
        ax.imshow(img, origin='lower', extent=extent, aspect='equal', interpolation='nearest')

        # Add certainty contours
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))
        contour = ax.contour(X, Y, certainty_grid, levels=[0.5, 0.75, 0.9],
                            colors='white', linewidths=1.5, alpha=0.6, linestyles='--')
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f%% certain')

        # Add gridlines
        for i in range(len(x_vals) + 1):
            ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for i in range(len(y_vals) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nDominant Outcome + Certainty', fontsize=12, fontweight='bold')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

# Add custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=magma_cmap(0.2), label='PASS (dominant)'),
    Patch(facecolor=magma_cmap(0.5), label='WEEP (dominant)'),
    Patch(facecolor=magma_cmap(0.9), label='FLOOD (dominant)'),
    Patch(facecolor=[0.7, 0.7, 0.7], label='Low certainty (faded)')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

fig.suptitle('Option 2: Three-Class Probability Heatmap\n(Dominant outcome with certainty intensity)',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path_2 = output_dir / 'option2_certainty_heatmap.png'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_2}")
plt.close()

# ===================================================================
# OPTION 3: Hybrid Gradient + Scatter + Boundaries
# Combines gradient background, scatter points, and decision boundaries
# ===================================================================
print("\n" + "=" * 80)
print("OPTION 3: Hybrid Gradient + Scatter + Boundaries")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (var1, var2) in enumerate(hydraulic_pairs):
    if var1 in df_full.columns and var2 in df_full.columns:
        ax = axes[idx]

        # LAYER 1: Gradient background (failure probability)
        df_full['is_failure'] = (df_full['DESC'] == 'WEEP') | (df_full['DESC'] == 'FLOOD')
        grouped = df_full.groupby([var1, var2])['is_failure'].mean() * 100

        # Create pivot table - discrete grid
        pivot = grouped.reset_index().pivot(index=var2, columns=var1, values='is_failure')

        # Get actual discrete values
        x_vals = sorted(df_full[var1].unique())
        y_vals = sorted(df_full[var2].unique())

        # Create coordinate arrays
        X, Y = np.meshgrid(np.arange(len(x_vals)), np.arange(len(y_vals)))

        # Define smooth gradient levels
        levels = np.linspace(0, 100, 21)

        # Filled contour plot - gradient background
        contourf = ax.contourf(X, Y, pivot.values, levels=levels,
                               cmap='magma', alpha=0.6, extend='neither')

        # LAYER 2: Scatter points (semi-transparent)
        for desc_val, color in DESC_COLORS.items():
            subset = df_full[df_full['DESC'] == desc_val]
            # Map to index coordinates
            x_indices = [x_vals.index(v) for v in subset[var1]]
            y_indices = [y_vals.index(v) for v in subset[var2]]

            ax.scatter(x_indices, y_indices, c=[color], alpha=0.3, s=25,
                      edgecolors='none', label=desc_val)

        # LAYER 3: Decision boundaries
        X_pair = df_full[[var1, var2]].values
        y_desc = df_full['DESC'].map({'PASS': 0, 'WEEP': 1, 'FLOOD': 2}).values

        # Train XGBoost classifier
        clf = XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        clf.fit(X_pair, y_desc)

        # Create meshgrid in actual value space
        x_min, x_max = df_full[var1].min(), df_full[var1].max()
        y_min, y_max = df_full[var2].min(), df_full[var2].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05

        xx, yy = np.meshgrid(
            np.linspace(x_min - x_padding, x_max + x_padding, 200),
            np.linspace(y_min - y_padding, y_max + y_padding, 200)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Map back to index coordinates for plotting
        xx_idx = np.interp(xx, x_vals, np.arange(len(x_vals)))
        yy_idx = np.interp(yy, y_vals, np.arange(len(y_vals)))

        # Draw decision boundaries
        ax.contour(xx_idx, yy_idx, Z, levels=[0.5, 1.5],
                  colors=['red', 'orange'], linewidths=2.5,
                  linestyles='--', alpha=0.9)

        # Clean up temporary column
        df_full.drop('is_failure', axis=1, inplace=True)

        # Set ticks to discrete values
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in x_vals], rotation=45, ha='right')
        ax.set_yticklabels([f'{int(v)}' if v == int(v) else f'{v:g}' for v in y_vals])

        ax.set_xlabel(var1, fontsize=12, fontweight='bold')
        ax.set_ylabel(var2, fontsize=12, fontweight='bold')
        ax.set_title(f'{var1} vs {var2}\nGradient + Data + Boundaries', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9, framealpha=0.8)

        # Colorbar for gradient
        cbar = plt.colorbar(contourf, ax=ax, alpha=0.6)
        cbar.set_label('Failure Rate (%)', fontsize=10)

fig.suptitle('Option 3: Hybrid Gradient Visualization\n(Gradient background + Scatter points + Decision boundaries)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
output_path_3 = output_dir / 'option3_hybrid.png'
plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_3}")
plt.close()

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("HYDRAULIC GRADIENT VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("\nGenerated 3 images (each with 6 gradient-style plots):")
print(f"  1. option1_probability_gradient.png")
print("     → Pure smooth gradient showing 0-100% failure risk")
print(f"  2. option2_certainty_heatmap.png")
print("     → Dominant outcome colored by certainty (faded = uncertain)")
print(f"  3. option3_hybrid.png")
print("     → Gradient background + scatter points + decision boundaries")
print("\nVisualization styles:")
print("  - OPTION 1: Most gradient-like (pure smooth surface)")
print("  - OPTION 2: Shows uncertainty/confidence visually")
print("  - OPTION 3: Comprehensive (combines all techniques)")
print("\n🎨 Whatever the professor wants - created for Karisa! 🎨")
print("=" * 80)
