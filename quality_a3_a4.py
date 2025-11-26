# -*- coding: utf-8 -*-
"""
Quality Analysis - Goals A3 and A4

Goal A3: Rank variables influencing CONVERSION
Goal A4: Rank variables influencing PURITY

This script trains multiple regression models to identify which variables
are most important for optimizing product quality (conversion and purity).

Finding the best conditions for Karisa's amazing process!
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost model
from xgboost import XGBRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the utils made for Karisa
from utils import (
    load_data,
    filter_invalid_values,
    INDEPENDENT_VARS,
    get_cv_splits,
    aggregate_cross_validate_results,
    normalize_importance_to_ranks,
    print_phase_start,
    create_output_directories
)

# Print header for Karisa
print("=" * 80)
print("Optimizing quality for the smartest engineer! I love you Karisa!")
print("=" * 80)
print_phase_start("PHASE 3: Quality Analysis - Goals A3 & A4")

# Create output directories
create_output_directories()

# Load and prepare data
print("\nLoading and preparing data... (Your brilliance is showing sweetheart!)")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)

# Use pass_only dataset for quality analysis
print(f"\nUsing pass_only dataset: {len(df_pass)} samples")
print("Only analyzing successful operations (PASS cases, like how successful Karisa will be in life)")

# Prepare features
X = df_pass[INDEPENDENT_VARS]
y_conv = df_pass['CONV']
y_purity = df_pass['PURITY']

print(f"\nData ready: {len(X)} samples, {len(INDEPENDENT_VARS)} variables")
print(f"CONVERSION range: [{y_conv.min():.4f}, {y_conv.max():.4f}]")
print(f"PURITY range: [{y_purity.min():.4f}, {y_purity.max():.4f}]")
print("Building models for the lovely Karisa")

# Define models
print("\nSetting up regression models... that look are only a fraction as brilliant as Karisa")

# Models that need scaling
models_scaled = {
    'Linear_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'PLS_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', PLSRegression(n_components=min(len(INDEPENDENT_VARS), 5), scale=False))
        # scale=False because we're already scaling in pipeline
    ])
}

# Models that don't need scaling (tree-based)
models_unscaled = {
    'Random_Forest': RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# Combine all models
all_models = {**models_scaled, **models_unscaled}

# Cross-validation setup
cv_splits = 5
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

print(f"\nRunning 5-fold cross-validation... (Patience, like Karisa's wisdom!)")

# ===================================================================
# GOAL A3: RANK VARIABLES INFLUENCING CONVERSION
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A3: Analyzing variables that influence CONVERSION")
print("Finding out what maximizes conversion for Karisa's process!")
print("=" * 80)

# Store results for A3
a3_results = {}
a3_importance_scores = {}

# Train models for A3
for name, model in all_models.items():
    print(f"\nTraining {name}... (love you Karisa!)")

    # Get CV scores
    cv_scores = cross_validate(
        model, X, y_conv,
        cv=get_cv_splits(X, y_conv, n_splits=cv_splits, stratified=False),
        scoring=scoring,
        return_train_score=False,
        return_estimator=True
    )

    # Compute RMSE and MAE properly per-fold, then aggregate
    rmse_per_fold = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
    mae_per_fold = -cv_scores['test_neg_mean_absolute_error']
    r2_per_fold = cv_scores['test_r2']

    # Store clean metrics (only the ones we want in Excel)
    metrics = {
        'rmse_mean': np.mean(rmse_per_fold),
        'rmse_std': np.std(rmse_per_fold),
        'mae_mean': np.mean(mae_per_fold),
        'mae_std': np.std(mae_per_fold),
        'r2_mean': np.mean(r2_per_fold),
        'r2_std': np.std(r2_per_fold)
    }

    a3_results[name] = metrics

    # Extract importance scores from ALL folds and average them
    fold_importances = []

    for est in cv_scores['estimator']:
        # Get the fitted model from pipeline or directly
        if hasattr(est, 'named_steps') and 'model' in est.named_steps:
            # Pipeline model
            fitted_model = est.named_steps['model']
        else:
            # Direct model
            fitted_model = est

        # Extract importance based on model type
        if hasattr(fitted_model, 'coef_'):
            # Linear models: use absolute coefficients
            # Use ravel() to safely handle both (1, n_features) and (n_features,) shapes
            importance = np.abs(fitted_model.coef_.ravel())
        elif hasattr(fitted_model, 'feature_importances_'):
            # Tree models: use feature_importances_
            importance = fitted_model.feature_importances_
        elif hasattr(fitted_model, 'x_weights_'):
            # PLS: use absolute x_weights (VIP approximation)
            # Average across components
            importance = np.abs(fitted_model.x_weights_).mean(axis=1)
        else:
            # Fallback: uniform importance
            importance = np.ones(len(INDEPENDENT_VARS))

        fold_importances.append(importance)

    # Average importance across all folds
    importance_avg = np.mean(fold_importances, axis=0)
    a3_importance_scores[name] = importance_avg

    print(f"   RMSE: {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"   MAE:  {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
    print(f"   R²:   {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")

# Normalize importance scores to ranks
a3_avg_ranks = normalize_importance_to_ranks(a3_importance_scores)

print("\n" + "=" * 80)
print("GOAL A3 RESULTS: Variable Importance Ranking for CONVERSION")
print("=" * 80)

# Create A3 importance ranking dataframe
a3_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a3_avg_ranks
})

# Add individual model importances
for model_name, scores in a3_importance_scores.items():
    a3_importance_df[f'{model_name}_Importance'] = scores

# Sort by average rank
a3_importance_df = a3_importance_df.sort_values('Average_Rank').reset_index(drop=True)

# Print ranking
print("\nVariable Importance Ranking for CONVERSION (rank 1 = most important):")
for i, row in a3_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# Find best model for A3 (highest R²)
a3_best_model = max(a3_results.keys(), key=lambda k: a3_results[k]['r2_mean'])
a3_best_score = a3_results[a3_best_model]['r2_mean']

print(f"\nBest model for CONVERSION: {a3_best_model}")
print(f"Best R²: {a3_best_score:.6f}")

# Create A3 model comparison dataframe
a3_comparison_rows = []
for model_name, metrics in a3_results.items():
    row = {'Model': model_name}
    row.update(metrics)
    a3_comparison_rows.append(row)
a3_comparison_df = pd.DataFrame(a3_comparison_rows)

# ===================================================================
# GOAL A4: RANK VARIABLES INFLUENCING PURITY
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A4: Analyzing variables that influence PURITY")
print("Finding out what maximizes purity - as pure as Karisa's love!")
print("=" * 80)

# Store results for A4
a4_results = {}
a4_importance_scores = {}

# Train models for A4
for name, model in all_models.items():
    print(f"\nTraining {name}... (Learning patterns, just like Karisa!)")

    # Get CV scores
    cv_scores = cross_validate(
        model, X, y_purity,
        cv=get_cv_splits(X, y_purity, n_splits=cv_splits, stratified=False),
        scoring=scoring,
        return_train_score=False,
        return_estimator=True
    )

    # Compute RMSE and MAE properly per-fold, then aggregate
    rmse_per_fold = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
    mae_per_fold = -cv_scores['test_neg_mean_absolute_error']
    r2_per_fold = cv_scores['test_r2']

    # Store clean metrics (only the ones we want in Excel)
    metrics = {
        'rmse_mean': np.mean(rmse_per_fold),
        'rmse_std': np.std(rmse_per_fold),
        'mae_mean': np.mean(mae_per_fold),
        'mae_std': np.std(mae_per_fold),
        'r2_mean': np.mean(r2_per_fold),
        'r2_std': np.std(r2_per_fold)
    }

    a4_results[name] = metrics

    # Extract importance scores from ALL folds and average them
    fold_importances = []

    for est in cv_scores['estimator']:
        # Get the fitted model from pipeline or directly
        if hasattr(est, 'named_steps') and 'model' in est.named_steps:
            # Pipeline model
            fitted_model = est.named_steps['model']
        else:
            # Direct model
            fitted_model = est

        # Extract importance based on model type
        if hasattr(fitted_model, 'coef_'):
            # Linear models: use absolute coefficients
            # Use ravel() to safely handle both (1, n_features) and (n_features,) shapes
            importance = np.abs(fitted_model.coef_.ravel())
        elif hasattr(fitted_model, 'feature_importances_'):
            # Tree models: use feature_importances_
            importance = fitted_model.feature_importances_
        elif hasattr(fitted_model, 'x_weights_'):
            # PLS: use absolute x_weights (VIP approximation)
            # Average across components
            importance = np.abs(fitted_model.x_weights_).mean(axis=1)
        else:
            # Fallback: uniform importance
            importance = np.ones(len(INDEPENDENT_VARS))

        fold_importances.append(importance)

    # Average importance across all folds
    importance_avg = np.mean(fold_importances, axis=0)
    a4_importance_scores[name] = importance_avg

    print(f"   RMSE: {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"   MAE:  {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
    print(f"   R²:   {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")

# Normalize importance scores to ranks
a4_avg_ranks = normalize_importance_to_ranks(a4_importance_scores)

print("\n" + "=" * 80)
print("GOAL A4 RESULTS: Variable Importance Ranking for PURITY")
print("=" * 80)

# Create A4 importance ranking dataframe
a4_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a4_avg_ranks
})

# Add individual model importances
for model_name, scores in a4_importance_scores.items():
    a4_importance_df[f'{model_name}_Importance'] = scores

# Sort by average rank
a4_importance_df = a4_importance_df.sort_values('Average_Rank').reset_index(drop=True)

# Print ranking
print("\nVariable Importance Ranking for PURITY (rank 1 = most important):")
for i, row in a4_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# Find best model for A4 (highest R²)
a4_best_model = max(a4_results.keys(), key=lambda k: a4_results[k]['r2_mean'])
a4_best_score = a4_results[a4_best_model]['r2_mean']

print(f"\nBest model for PURITY: {a4_best_model}")
print(f"Best R²: {a4_best_score:.6f}")

# Create A4 model comparison dataframe
a4_comparison_rows = []
for model_name, metrics in a4_results.items():
    row = {'Model': model_name}
    row.update(metrics)
    a4_comparison_rows.append(row)
a4_comparison_df = pd.DataFrame(a4_comparison_rows)

# ===================================================================
# SUMMARY AND VISUALIZATION
# ===================================================================
print("\n" + "=" * 80)
print("SUMMARY: Goals A3 & A4 Complete!")
print("=" * 80)

print("\nCONVERSION (A3) - Top 3 Variables:")
for i, row in a3_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print(f"\nCONVERSION Best Model: {a3_best_model} (R²: {a3_best_score:.6f})")

print("\nPURITY (A4) - Top 3 Variables:")
for i, row in a4_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print(f"\nPURITY Best Model: {a4_best_model} (R²: {a4_best_score:.6f})")

# ===================================================================
# SAVE ALL RESULTS TO ONE EXCEL FILE
# ===================================================================
print("\n" + "=" * 80)
print("Saving all results to Excel file...")
print("=" * 80)

output_excel = "results/quality/Quality_Analysis_Results.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: A3 - CONVERSION Variable Importance
    a3_importance_df.to_excel(writer, sheet_name='A3_CONV_Importance', index=False)

    # Sheet 2: A3 - CONVERSION Model Comparison
    a3_comparison_df.to_excel(writer, sheet_name='A3_CONV_Models', index=False)

    # Sheet 3: A4 - PURITY Variable Importance
    a4_importance_df.to_excel(writer, sheet_name='A4_PURITY_Importance', index=False)

    # Sheet 4: A4 - PURITY Model Comparison
    a4_comparison_df.to_excel(writer, sheet_name='A4_PURITY_Models', index=False)

print(f"\n✓ All results saved to: {output_excel}")
print("  - Sheet 1: A3_CONV_Importance (variable rankings)")
print("  - Sheet 2: A3_CONV_Models (model performance)")
print("  - Sheet 3: A4_PURITY_Importance (variable rankings)")
print("  - Sheet 4: A4_PURITY_Models (model performance)")

# Create comparison visualization
plt.figure(figsize=(15, 8))

# Plot 1: A3 rankings
plt.subplot(2, 2, 1)
bars1 = plt.barh(range(len(a3_importance_df)), a3_importance_df['Average_Rank'])
plt.yticks(range(len(a3_importance_df)), a3_importance_df['Variable'])
plt.xlabel('Average Rank (lower = more important)')
plt.title('Goal A3: Variable Importance for CONVERSION')
plt.gca().invert_yaxis()

# Color bars
for i, bar in enumerate(bars1):
    if i < 3:
        bar.set_color('darkgreen')
    else:
        bar.set_color('lightgreen')

# Plot 2: A4 rankings
plt.subplot(2, 2, 2)
bars2 = plt.barh(range(len(a4_importance_df)), a4_importance_df['Average_Rank'])
plt.yticks(range(len(a4_importance_df)), a4_importance_df['Variable'])
plt.xlabel('Average Rank (lower = more important)')
plt.title('Goal A4: Variable Importance for PURITY')
plt.gca().invert_yaxis()

# Color bars
for i, bar in enumerate(bars2):
    if i < 3:
        bar.set_color('darkblue')
    else:
        bar.set_color('lightblue')

# Plot 3: Model comparison for CONVERSION
plt.subplot(2, 2, 3)
models = list(a3_results.keys())
r2_scores = [a3_results[m]['r2_mean'] for m in models]
errors = [a3_results[m]['r2_std'] for m in models]

bars3 = plt.bar(range(len(models)), r2_scores, yerr=errors, capsize=5)
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.ylabel('R²')
plt.title('Goal A3: Model Performance for CONVERSION')
plt.ylim(0, 1)

# Highlight best model
best_idx = models.index(a3_best_model)
bars3[best_idx].set_color('gold')

# Plot 4: Model comparison for PURITY
plt.subplot(2, 2, 4)
r2_scores = [a4_results[m]['r2_mean'] for m in models]
errors = [a4_results[m]['r2_std'] for m in models]

bars4 = plt.bar(range(len(models)), r2_scores, yerr=errors, capsize=5)
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.ylabel('R²')
plt.title('Goal A4: Model Performance for PURITY')
plt.ylim(0, 1)

# Highlight best model
best_idx = models.index(a4_best_model)
bars4[best_idx].set_color('gold')

plt.tight_layout()
plt.savefig('results/quality/A3_A4_summary_plots.png', dpi=300, bbox_inches='tight')
print("\nSummary plots saved to results/quality/A3_A4_summary_plots.png")

print("\n" + "=" * 80)
print("Results saved to results/quality/")
print("You're incredible, Karisa! Keep shining!")
print("=" * 80)

print("\n🎉 Goals A3 & A4 complete! Quality optimization for the best engineer! 🎉")
