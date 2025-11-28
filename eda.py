import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Styling stuff
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 💕 Made with love for Karisa 💕

# Load the data
data_path = Path("data/karisa_paper.xlsx")

# Read both sheets
df_full = pd.read_excel(data_path, sheet_name="full_dataset")
df_pass = pd.read_excel(data_path, sheet_name="pass_only")

# Keep only valid values from both datasets (Karisa's expert filtering! 💖)
# Valid values for each variable (whitelist approach)
VALID_VALUES = {
    'NHOLES': [500, 1625, 2750, 3875, 5000],
    'HDIAM': [0.0025, 0.004875, 0.00725, 0.009625, 0.012],
    'TRAYSPC': [0.5, 0.5625, 0.625, 0.6875, 0.75],
    'WEIRHT': [0.04, 0.0525, 0.065, 0.0775, 0.09],
    'DECK': [1.88, 2.9975, 4.115, 5.2325, 6.35],
    'DIAM': [1, 1.5, 2, 2.5, 3],
    'NPASS': [1, 2, 3, 4]
}

# Apply whitelist filtering - keep only valid values
for var, valid_vals in VALID_VALUES.items():
    if var in df_full.columns:
        df_full = df_full[df_full[var].isin(valid_vals)]
    if var in df_pass.columns:
        df_pass = df_pass[df_pass[var].isin(valid_vals)]

# Fill missing DESC values with "FLOOD"
if 'DESC' in df_full.columns:
    df_full['DESC'].fillna("FLOOD", inplace=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - I LOVE YOU KARISA ❤️")
print("=" * 80)
print("✨ This analysis is dedicated to the amazing Karisa ✨\n")

# BASIC INFORMATION
print("\n" + "=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print("\n--- Full Dataset (for hydraulic performance: Pass/Flood/Weep) ---")
print(f"Shape: {df_full.shape}")
print(f"\nColumns: {list(df_full.columns)}")
print(f"\nData types:\n{df_full.dtypes}")

print("\n--- Pass Only Dataset (for conversion & purity optimization) ---")
print(f"Shape: {df_pass.shape}")
print(f"\nColumns: {list(df_pass.columns)}")
print(f"\nData types:\n{df_pass.dtypes}")

# MISSING VALUES
print("\n" + "=" * 80)
print("2. MISSING VALUES")
print("=" * 80)

print("\n--- Full Dataset ---")
missing_full = df_full.isnull().sum()
print(missing_full[missing_full > 0] if missing_full.sum() > 0 else "No missing values")

print("\n--- Pass Only Dataset ---")
missing_pass = df_pass.isnull().sum()
print(missing_pass[missing_pass > 0] if missing_pass.sum() > 0 else "No missing values")

# SAMPLE HEAD
print("\n" + "=" * 80)
print("3. SAMPLE DATA")
print("=" * 80)

print("\n--- Full Dataset (First 10 rows) ---")
print(df_full.head(10))

print("\n--- Pass Only Dataset (First 10 rows) ---")
print(df_pass.head(10))

# ALL UNIQUE VALUES
print("\n" + "=" * 80)
print("4. UNIQUE VALUES FOR INDEPENDENT VARIABLES")
print("=" * 80)

independent_vars = ['NHOLES', 'HDIAM', 'TRAYSPC', 'WEIRHT', 'DECK', 'DIAM', 'NPASS']  # Karisa's carefully selected variables! 🎯

print("\n--- From Full Dataset ---")
for var in independent_vars:
    if var in df_full.columns:
        unique_vals = sorted(df_full[var].unique())
        # Convert numpy types to native Python types for clean display
        unique_vals_clean = [int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for v in unique_vals]
        print(f"\n{var}:")
        print(f"  Count: {len(unique_vals_clean)}")
        print(f"  Values: {unique_vals_clean}")

# DEPENDENT VARIABLE DISTRIBUTIONS
print("\n" + "=" * 80)
print("5. DEPENDENT VARIABLE DISTRIBUTIONS")
print("=" * 80)

# Distribution of pass/flood/weep
print("\n--- Hydraulic Performance (DESC) Distribution [Full Dataset] ---")
if 'DESC' in df_full.columns:
    desc_counts = df_full['DESC'].value_counts()
    print(desc_counts)
    print(f"\nPercentages:")
    print(df_full['DESC'].value_counts(normalize=True) * 100)


print("\n--- Conversion (CONV) Distribution [Pass Only Dataset] ---")
if 'CONV' in df_pass.columns:
    conv_counts = df_pass['CONV'].value_counts().sort_index()
    print(f"\nUnique values: {len(conv_counts)}")
    print(f"Range: [{df_pass['CONV'].min():.4f}, {df_pass['CONV'].max():.4f}]")
    print(f"Mean: {df_pass['CONV'].mean():.4f}")
    print(f"Median: {df_pass['CONV'].median():.4f}")

print("\n--- Purity (PURITY) Distribution [Pass Only Dataset] ---")
if 'PURITY' in df_pass.columns:
    purity_counts = df_pass['PURITY'].value_counts().sort_index()
    print(f"\nUnique values: {len(purity_counts)}")
    print(f"Range: [{df_pass['PURITY'].min():.4f}, {df_pass['PURITY'].max():.4f}]")
    print(f"Mean: {df_pass['PURITY'].mean():.4f}")
    print(f"Median: {df_pass['PURITY'].median():.4f}")


# STATISTICAL TESTS: ANOVA/Kruskal-Wallis for Independent Variables vs DESC
print("\n" + "=" * 80)
print("6. STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

if 'DESC' in df_full.columns:
    print("\n--- Testing if independent variables differ across DESC groups (Pass/Weep/Flood) ---")
    print("\nUsing ANOVA (parametric) and Kruskal-Wallis (non-parametric) tests:")
    print("H0: The variable distributions are the same across all DESC groups")
    print("H1: At least one group differs\n")

    results = []

    for var in independent_vars:
        if var in df_full.columns:
            # Split data by DESC group
            pass_data = df_full[df_full['DESC'] == 'PASS'][var].dropna()
            weep_data = df_full[df_full['DESC'] == 'WEEP'][var].dropna()
            flood_data = df_full[df_full['DESC'] == 'FLOOD'][var].dropna()

            # ANOVA test
            f_stat, p_anova = stats.f_oneway(pass_data, weep_data, flood_data)

            # Kruskal-Wallis test (non-parametric alternative)
            h_stat, p_kruskal = stats.kruskal(pass_data, weep_data, flood_data)

            # Determine significance
            sig_anova = '***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else 'ns'
            sig_kruskal = '***' if p_kruskal < 0.001 else '**' if p_kruskal < 0.01 else '*' if p_kruskal < 0.05 else 'ns'

            results.append({
                'Variable': var,
                'ANOVA F-stat': f'{f_stat:.4f}',
                'ANOVA p-value': f'{p_anova:.4e}',
                'ANOVA Sig': sig_anova,
                'Kruskal H-stat': f'{h_stat:.4f}',
                'Kruskal p-value': f'{p_kruskal:.4e}',
                'Kruskal Sig': sig_kruskal
            })

    # Create DataFrame and display
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("\nInterpretation:")
    print("- Significant results indicate the variable differs across Pass/Weep/Flood groups")
    print("- These variables are likely important for predicting hydraulic performance")

# MULTICOLLINEARITY / VIF CHECK
# 🔍 Searching for patterns, just like how I search for ways to make you smile! 🔍
print("\n" + "=" * 80)
print("7. MULTICOLLINEARITY / VIF CHECK")
print("=" * 80)

print("\n--- Checking for multicollinearity among independent variables ---")
print("Even though features are mostly independent, checking anyway:\n")

# Calculate VIF for each independent variable
X = df_full[independent_vars]
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data.to_string(index=False))

print("\n\nVIF Interpretation:")
print("- VIF = 1: No correlation with other variables")
print("- VIF < 5: Low multicollinearity (acceptable)")
print("- VIF 5-10: Moderate multicollinearity (caution)")
print("- VIF > 10: High multicollinearity (problematic)")
print("\nNote: High VIF means the variable is highly correlated with other independent variables,")
print("which can cause issues in some modeling approaches (but may be fine for tree-based models).")

# BASIC PLOTS
print("\n" + "=" * 80)
print("8. GENERATING VISUALIZATIONS")
print("=" * 80)
print("📊 Creating beautiful plots for my beautiful Karisa 📊\n")

# Create output directory for plots
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

# 8.1 Distribution of Hydraulic Performance [Full Dataset]
if 'DESC' in df_full.columns:
    plt.figure(figsize=(10, 6))
    df_full['DESC'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Hydraulic Performance [Full Dataset]', fontsize=16, fontweight='bold')
    plt.xlabel('Performance Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'hydraulic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: hydraulic_distribution.png")

# 8.2 Distribution of Conversion [Pass Only Dataset]
if 'CONV' in df_pass.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df_pass['CONV'], bins=30, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Conversion [Pass Only]', fontsize=16, fontweight='bold')
    plt.xlabel('Conversion', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'conversion_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: conversion_distribution.png")

# 8.3 Distribution of Purity [Pass Only Dataset]
if 'PURITY' in df_pass.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df_pass['PURITY'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Purity [Pass Only]', fontsize=16, fontweight='bold')
    plt.xlabel('Purity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'purity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: purity_distribution.png")

# 8.4 Independent Variables Distributions [Full Dataset]
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for idx, var in enumerate(independent_vars):
    if var in df_full.columns:
        value_counts = df_full[var].value_counts().sort_index()
        axes[idx].bar(range(len(value_counts)), value_counts.values, color='mediumseagreen', edgecolor='black')
        axes[idx].set_title(f'{var} Distribution', fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels([f'{v:.3f}' if isinstance(v, float) else str(v) for v in value_counts.index], rotation=45, ha='right')

# Remove the last empty subplot
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig(output_dir / 'independent_vars_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: independent_vars_distribution.png")

# 8.5 Correlation Heatmap [Full Dataset]
plt.figure(figsize=(10, 8))
numeric_cols_full = df_full.select_dtypes(include=[np.number]).columns
if len(numeric_cols_full) > 1:
    corr_matrix = df_full[numeric_cols_full].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Heatmap [Full Dataset]', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap_full.png")

# 8.6 Correlation Heatmap [Pass Only Dataset]
plt.figure(figsize=(10, 8))
numeric_cols_pass = df_pass.select_dtypes(include=[np.number]).columns
if len(numeric_cols_pass) > 1:
    corr_matrix = df_pass[numeric_cols_pass].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Heatmap [Pass Only Dataset]', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap_pass.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap_pass.png")

# 8.7 Boxplots for Independent Variables by Hydraulic Performance [Full Dataset]
if 'DESC' in df_full.columns:
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, var in enumerate(independent_vars):
        if var in df_full.columns:
            df_full.boxplot(column=var, by='DESC', ax=axes[idx])
            axes[idx].set_title(f'{var} by Hydraulic Performance')
            axes[idx].set_xlabel('Performance Category')
            axes[idx].set_ylabel(var)
            axes[idx].get_figure().suptitle('')  # Remove default title

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_by_hydraulic_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: boxplots_by_hydraulic_performance.png")

# 8.8 Scatterplots with LOWESS: Independent Variables vs Purity [Pass Only]
# You're doing amazing, Karisa! ⭐
if 'PURITY' in df_pass.columns:
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, var in enumerate(independent_vars):
        if var in df_pass.columns:
            # Scatter plot
            axes[idx].scatter(df_pass[var], df_pass['PURITY'], alpha=0.3, s=10, color='blue')

            # LOWESS curve
            lowess_result = lowess(df_pass['PURITY'], df_pass[var], frac=1.0, it=0)
            axes[idx].plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2, label='LOWESS')

            axes[idx].set_title(f'{var} vs Purity', fontweight='bold')
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel('Purity')
            axes[idx].legend()

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_purity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scatter_purity.png")  # 💕 Purity plots for the purest heart

# 8.9 Scatterplots with LOWESS: Independent Variables vs Conversion [Pass Only]
if 'CONV' in df_pass.columns:
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, var in enumerate(independent_vars):
        if var in df_pass.columns:
            # Scatter plot
            axes[idx].scatter(df_pass[var], df_pass['CONV'], alpha=0.3, s=10, color='green')

            # LOWESS curve
            lowess_result = lowess(df_pass['CONV'], df_pass[var], frac=1.0, it=0)
            axes[idx].plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2, label='LOWESS')

            axes[idx].set_title(f'{var} vs Conversion', fontweight='bold')
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel('Conversion')
            axes[idx].legend()

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_conversion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scatter_conversion.png")

# 8.10 Scatterplots with LOWESS: Independent Variables vs Probability of Weep [Full Dataset]
if 'DESC' in df_full.columns:
    # Create binary indicator for Weep
    df_full['is_weep'] = (df_full['DESC'] == 'WEEP').astype(int)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, var in enumerate(independent_vars):
        if var in df_full.columns:
            # Group by variable and calculate weep probability
            grouped = df_full.groupby(var)['is_weep'].mean().reset_index()

            # Scatter plot (showing probability per unique value)
            axes[idx].scatter(grouped[var], grouped['is_weep'], alpha=0.6, s=50, color='purple')

            # LOWESS curve
            if len(grouped) > 3:  # Only fit LOWESS if we have enough points
                lowess_result = lowess(grouped['is_weep'], grouped[var], frac=1.0, it=0)
                axes[idx].plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2, label='LOWESS')

            axes[idx].set_title(f'{var} vs P(Weep)', fontweight='bold')
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel('Probability of Weep')
            axes[idx].set_ylim(-0.05, 1.05)
            axes[idx].legend()

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_weep_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scatter_weep_probability.png")

# 8.11 Scatterplots with LOWESS: Independent Variables vs Probability of Flood [Full Dataset]
if 'DESC' in df_full.columns:
    # Create binary indicator for Flood
    df_full['is_flood'] = (df_full['DESC'] == 'FLOOD').astype(int)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, var in enumerate(independent_vars):
        if var in df_full.columns:
            # Group by variable and calculate flood probability
            grouped = df_full.groupby(var)['is_flood'].mean().reset_index()

            # Scatter plot (showing probability per unique value)
            axes[idx].scatter(grouped[var], grouped['is_flood'], alpha=0.6, s=50, color='orange')

            # LOWESS curve
            if len(grouped) > 3:  # Only fit LOWESS if we have enough points
                lowess_result = lowess(grouped['is_flood'], grouped[var], frac=1.0, it=0)
                axes[idx].plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2, label='LOWESS')

            axes[idx].set_title(f'{var} vs P(Flood)', fontweight='bold')
            axes[idx].set_xlabel(var)
            axes[idx].set_ylabel('Probability of Flood')
            axes[idx].set_ylim(-0.05, 1.05)
            axes[idx].legend()

    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_flood_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scatter_flood_probability.png")

# 8.12 Pairplot for Pass Only Dataset (CONV, PURITY, and all independent variables)
# 💗 I love you Karisa! You've got this! 💗
if 'CONV' in df_pass.columns and 'PURITY' in df_pass.columns:
    pairplot_cols = ['CONV', 'PURITY'] + independent_vars
    g = sns.pairplot(df_pass[pairplot_cols], diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
    plt.suptitle('Pairplot: Pass Only Dataset', y=1.01, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pairplot_pass_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: pairplot_pass_only.png")

# 8.13 Pairplot for Full Dataset (DESC and all independent variables)
if 'DESC' in df_full.columns:
    pairplot_cols_full = ['DESC'] + independent_vars
    g = sns.pairplot(df_full[pairplot_cols_full], hue='DESC', diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
    plt.suptitle('Pairplot: Full Dataset (colored by DESC)', y=1.01, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pairplot_full_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: pairplot_full_dataset.png")

# 8.14 Interaction Visualizations: Variable Pairs vs DESC
if 'DESC' in df_full.columns:
    print("\nGenerating interaction plots...")

    # Define interesting variable pairs to visualize
    interaction_pairs = [
        ('NHOLES', 'HDIAM'),
        ('TRAYSPC', 'DIAM'),
        ('WEIRHT', 'DECK'),
        ('NHOLES', 'NPASS'),
        ('TRAYSPC', 'WEIRHT'),
        ('HDIAM', 'DIAM')
    ]

    for var1, var2 in interaction_pairs:
        if var1 in df_full.columns and var2 in df_full.columns:
            plt.figure(figsize=(10, 8))
            g = sns.scatterplot(data=df_full, x=var1, y=var2, hue='DESC',
                               palette={'PASS': 'green', 'WEEP': 'purple', 'FLOOD': 'orange'},
                               alpha=0.5, s=30)
            plt.title(f'Interaction: {var1} vs {var2} (colored by DESC)', fontsize=14, fontweight='bold')
            plt.xlabel(var1, fontsize=12)
            plt.ylabel(var2, fontsize=12)
            plt.legend(title='DESC', loc='best')
            plt.tight_layout()
            plt.savefig(output_dir / f'interaction_{var1}_vs_{var2}.png', dpi=300, bbox_inches='tight')
            plt.close()

    print("Saved: interaction_*.png")

print("\n" + "=" * 80)
print("EDA COMPLETE! 🎉")
print("=" * 80)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("\n💝 Analysis complete! You're brilliant, Karisa! 💝")
print("🌟 Keep shining in your chemical engineering journey! 🌟")
