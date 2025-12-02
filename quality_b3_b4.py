# -*- coding: utf-8 -*-
"""
Quality Analysis - Goals B3, B4, and Combined

Goal B3: Find combinations with highest CONVERSION
Goal B4: Find combinations with highest PURITY
Goal Combined: Find optimal region (high in BOTH)

This script identifies the best operating conditions for maximizing
product quality (conversion and purity).

Finding the sweet spot for Karisa's amazing process!
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
    deduplicate_data,
    INDEPENDENT_VARS,
    get_cv_splits,
    aggregate_cross_validate_results,
    get_unique_combinations,
    print_phase_start
)

# Print header for Karisa
print("=" * 80)
print("Finding the optimal conditions! (For the best engineer ever!)")
print("=" * 80)
print_phase_start("PHASE 3: Quality Analysis - Goals B3, B4 & Combined")

# Load and prepare data
print("\nLoading and preparing data... (Your brilliance is showing sweetheart!)")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)

# Use pass_only dataset
print(f"\nUsing pass_only dataset: {len(df_pass)} samples")

# Prepare features
X = df_pass[INDEPENDENT_VARS]
y_conv = df_pass['CONV']
y_purity = df_pass['PURITY']

print(f"\nData ready: {len(X)} samples, {len(INDEPENDENT_VARS)} variables")

# Get unique combinations
print("\nExtracting unique combinations from dataset...")
unique_combos = get_unique_combinations(df_pass, INDEPENDENT_VARS)
print(f"Found {len(unique_combos)} unique operating combinations")

# Define models
print("\nSetting up regression models...")

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
    ])
}

# Models that don't need scaling
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

# ===================================================================
# GOAL B3: FIND COMBINATIONS WITH HIGHEST CONVERSION
# ===================================================================
print("\n" + "=" * 80)
print("GOAL B3: Finding combinations with highest CONVERSION")
print("Maximizing conversion for Karisa's process!")
print("=" * 80)

# Evaluate models using 5-fold CV with R²
print("\nEvaluating models with 5-fold CV...")
b3_results = {}

for name, model in all_models.items():
    print(f"  Evaluating {name}...")

    # Cross-validate
    cv_scores = cross_validate(
        model, X, y_conv,
        cv=get_cv_splits(X, y_conv, n_splits=cv_splits, stratified=False),
        scoring=scoring,
        return_train_score=False
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

    b3_results[name] = metrics

    print(f"    RMSE: {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"    MAE:  {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
    print(f"    R²:   {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")

# Select best model based on R²
best_b3_model_name = max(b3_results.keys(), key=lambda k: b3_results[k]['r2_mean'])
best_b3_r2 = b3_results[best_b3_model_name]['r2_mean']

print(f"\nSelected model for B3: {best_b3_model_name} (R²: {best_b3_r2:.6f})")

# Retrain best model on FULL dataset for final predictions
print(f"Retraining {best_b3_model_name} on full dataset for predictions...")
best_b3_model = all_models[best_b3_model_name]
best_b3_model.fit(X, y_conv)

# Predict CONVERSION for all unique combinations
print(f"\nPredicting CONVERSION for {len(unique_combos)} unique combinations...")
conv_predictions = best_b3_model.predict(unique_combos)

# Add predictions to dataframe
b3_results_df = unique_combos.copy()
b3_results_df['Predicted_CONV'] = conv_predictions

# Categorize into performance zones using percentiles
p90 = np.percentile(conv_predictions, 90)
p70 = np.percentile(conv_predictions, 70)

def categorize_performance_conv(pred_conv):
    if pred_conv >= p90:
        return f'Excellent (Top 10%, ≥{p90:.4f})'
    elif pred_conv >= p70:
        return f'Good (70-90%, ≥{p70:.4f})'
    else:
        return f'Average (< 70%)'

b3_results_df['Performance_Category'] = b3_results_df['Predicted_CONV'].apply(categorize_performance_conv)

# Sort by prediction (highest first)
b3_results_df = b3_results_df.sort_values('Predicted_CONV', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("GOAL B3 RESULTS: CONVERSION Performance Assessment")
print("=" * 80)

# Show summary statistics for each zone
print("\nPerformance Zone Summary:")
for category in [f'Excellent (Top 10%, ≥{p90:.4f})', f'Good (70-90%, ≥{p70:.4f})', 'Average (< 70%)']:
    count = (b3_results_df['Performance_Category'] == category).sum()
    pct = count / len(b3_results_df) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

# Show examples from each zone
print("\n" + "-" * 80)
print(f"EXCELLENT COMBINATIONS (Top 10%, Predicted CONV ≥ {p90:.4f}):")
print("-" * 80)
excellent = b3_results_df[b3_results_df['Performance_Category'] == f'Excellent (Top 10%, ≥{p90:.4f})']
if len(excellent) > 0:
    print(f"\nShowing top 5 of {len(excellent)} excellent combinations:")
    for i, row in excellent.head(5).iterrows():
        print(f"\n{i+1}. Predicted CONV = {row['Predicted_CONV']:.4f}")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No excellent combinations found.")

print("\n" + "-" * 80)
print(f"GOOD COMBINATIONS (70-90%, Predicted CONV ≥ {p70:.4f}):")
print("-" * 80)
good = b3_results_df[b3_results_df['Performance_Category'] == f'Good (70-90%, ≥{p70:.4f})']
if len(good) > 0:
    print(f"\nShowing 3 examples of {len(good)} good combinations:")
    for i, row in good.head(3).iterrows():
        print(f"\nPredicted CONV = {row['Predicted_CONV']:.4f}")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No good combinations found.")

# ===================================================================
# GOAL B4: FIND COMBINATIONS WITH HIGHEST PURITY
# ===================================================================
print("\n" + "=" * 80)
print("GOAL B4: Finding combinations with highest PURITY")
print("Maximizing purity for Karisa's excellence!")
print("=" * 80)

# Evaluate models using 5-fold CV with R²
print("\nEvaluating models with 5-fold CV...")
b4_results = {}

for name, model in all_models.items():
    print(f"  Evaluating {name}...")

    # Cross-validate
    cv_scores = cross_validate(
        model, X, y_purity,
        cv=get_cv_splits(X, y_purity, n_splits=cv_splits, stratified=False),
        scoring=scoring,
        return_train_score=False
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

    b4_results[name] = metrics

    print(f"    RMSE: {metrics['rmse_mean']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"    MAE:  {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
    print(f"    R²:   {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")

# Select best model based on R²
best_b4_model_name = max(b4_results.keys(), key=lambda k: b4_results[k]['r2_mean'])
best_b4_r2 = b4_results[best_b4_model_name]['r2_mean']

print(f"\nSelected model for B4: {best_b4_model_name} (R²: {best_b4_r2:.6f})")

# Retrain best model on FULL dataset for final predictions
print(f"Retraining {best_b4_model_name} on full dataset for predictions...")
best_b4_model = all_models[best_b4_model_name]
best_b4_model.fit(X, y_purity)

# Predict PURITY for all unique combinations
print(f"\nPredicting PURITY for {len(unique_combos)} unique combinations...")
purity_predictions = best_b4_model.predict(unique_combos)

# Add predictions to dataframe
b4_results_df = unique_combos.copy()
b4_results_df['Predicted_PURITY'] = purity_predictions

# Categorize into performance zones using percentiles
p90_purity = np.percentile(purity_predictions, 90)
p70_purity = np.percentile(purity_predictions, 70)

def categorize_performance_purity(pred_purity):
    if pred_purity >= p90_purity:
        return f'Excellent (Top 10%, ≥{p90_purity:.4f})'
    elif pred_purity >= p70_purity:
        return f'Good (70-90%, ≥{p70_purity:.4f})'
    else:
        return f'Average (< 70%)'

b4_results_df['Performance_Category'] = b4_results_df['Predicted_PURITY'].apply(categorize_performance_purity)

# Sort by prediction (highest first)
b4_results_df = b4_results_df.sort_values('Predicted_PURITY', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("GOAL B4 RESULTS: PURITY Performance Assessment")
print("=" * 80)

# Show summary statistics for each zone
print("\nPerformance Zone Summary:")
for category in [f'Excellent (Top 10%, ≥{p90_purity:.4f})', f'Good (70-90%, ≥{p70_purity:.4f})', 'Average (< 70%)']:
    count = (b4_results_df['Performance_Category'] == category).sum()
    pct = count / len(b4_results_df) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

# Show examples from each zone
print("\n" + "-" * 80)
print(f"EXCELLENT COMBINATIONS (Top 10%, Predicted PURITY ≥ {p90_purity:.4f}):")
print("-" * 80)
excellent = b4_results_df[b4_results_df['Performance_Category'] == f'Excellent (Top 10%, ≥{p90_purity:.4f})']
if len(excellent) > 0:
    print(f"\nShowing top 5 of {len(excellent)} excellent combinations:")
    for i, row in excellent.head(5).iterrows():
        print(f"\n{i+1}. Predicted PURITY = {row['Predicted_PURITY']:.4f}")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No excellent combinations found.")

print("\n" + "-" * 80)
print(f"GOOD COMBINATIONS (70-90%, Predicted PURITY ≥ {p70_purity:.4f}):")
print("-" * 80)
good = b4_results_df[b4_results_df['Performance_Category'] == f'Good (70-90%, ≥{p70_purity:.4f})']
if len(good) > 0:
    print(f"\nShowing 3 examples of {len(good)} good combinations:")
    for i, row in good.head(3).iterrows():
        print(f"\nPredicted PURITY = {row['Predicted_PURITY']:.4f}")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No good combinations found.")

# ===================================================================
# COMBINED ANALYSIS: OPTIMAL REGION (HIGH IN BOTH)
# ===================================================================
print("\n" + "=" * 80)
print("COMBINED ANALYSIS: Finding Optimal Region (High CONV AND PURITY)")
print("Finding the sweet spot for Karisa!")
print("=" * 80)

# Create combined dataframe with both predictions
combined_df = unique_combos.copy()
combined_df['Predicted_CONV'] = conv_predictions
combined_df['Predicted_PURITY'] = purity_predictions

# Calculate simple average combined score (not geometric mean to avoid compression)
combined_df['Combined_Score'] = (combined_df['Predicted_CONV'] + combined_df['Predicted_PURITY']) / 2

# Identify top 10% in BOTH metrics
top10_conv_threshold = np.percentile(conv_predictions, 90)
top10_purity_threshold = np.percentile(purity_predictions, 90)

combined_df['In_Top10_CONV'] = combined_df['Predicted_CONV'] >= top10_conv_threshold
combined_df['In_Top10_PURITY'] = combined_df['Predicted_PURITY'] >= top10_purity_threshold
combined_df['In_Optimal_Region'] = combined_df['In_Top10_CONV'] & combined_df['In_Top10_PURITY']

# Categorize combinations
def categorize_combined(row):
    if row['In_Optimal_Region']:
        return 'Optimal (Both Top 10%)'
    elif row['In_Top10_CONV'] or row['In_Top10_PURITY']:
        return 'Good (One Top 10%)'
    else:
        return 'Average'

combined_df['Overall_Category'] = combined_df.apply(categorize_combined, axis=1)

# Sort by combined score (highest first)
combined_df = combined_df.sort_values('Combined_Score', ascending=False).reset_index(drop=True)

print("\nCombined Performance Summary:")
for category in ['Optimal (Both Top 10%)', 'Good (One Top 10%)', 'Average']:
    count = (combined_df['Overall_Category'] == category).sum()
    pct = count / len(combined_df) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

# Get optimal region combinations
optimal_region = combined_df[combined_df['In_Optimal_Region'] == True]

print("\n" + "-" * 80)
print(f"OPTIMAL REGION COMBINATIONS (Top 10% in BOTH):")
print("-" * 80)
if len(optimal_region) > 0:
    print(f"\nFound {len(optimal_region)} optimal combinations!")
    print(f"\nTop 10 optimal combinations by combined score:")
    for i, row in optimal_region.head(10).iterrows():
        print(f"\n{i+1}. Combined Score = {row['Combined_Score']:.4f}")
        print(f"   Predicted CONV   = {row['Predicted_CONV']:.4f}")
        print(f"   Predicted PURITY = {row['Predicted_PURITY']:.4f}")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No combinations found in optimal region (top 10% for BOTH metrics).")
    print("This suggests trade-offs between CONV and PURITY.")

# ===================================================================
# SAVE RESULTS TO EXCEL FILE
# ===================================================================
print("\n" + "=" * 80)
print("Saving results to Excel file...")
print("=" * 80)

# Load existing Excel file and add new sheets
excel_path = "results/quality/Quality_Analysis_Results.xlsx"

try:
    # Try to load existing file
    with pd.ExcelFile(excel_path) as xls:
        existing_sheets = {sheet_name: pd.read_excel(xls, sheet_name) for sheet_name in xls.sheet_names}
    print(f"Loaded existing file: {excel_path}")
except FileNotFoundError:
    existing_sheets = {}
    print(f"Creating new file: {excel_path}")

# Write all sheets (existing + new)
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Write existing sheets first
    for sheet_name, df in existing_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Add new sheets for B3, B4, and Combined
    b3_results_df.to_excel(writer, sheet_name='B3_CONV_Performance', index=False)
    b4_results_df.to_excel(writer, sheet_name='B4_PURITY_Performance', index=False)
    combined_df.to_excel(writer, sheet_name='B3_B4_Optimal_Region', index=False)

print(f"\n✓ Results saved to: {excel_path}")
print("  - Sheet 5: B3_CONV_Performance (all combinations with CONV predictions)")
print("  - Sheet 6: B4_PURITY_Performance (all combinations with PURITY predictions)")
print("  - Sheet 7: B3_B4_Optimal_Region (combined analysis with optimal region)")

# ===================================================================
# VISUALIZATION: CONV vs PURITY SCATTER PLOT
# ===================================================================
print("\n" + "=" * 80)
print("Creating visualization...")
print("=" * 80)

plt.figure(figsize=(12, 10))

# Color map for categories
color_map = {
    'Optimal (Both Top 10%)': 'gold',
    'Good (One Top 10%)': 'lightblue',
    'Average': 'lightgray'
}

for category in ['Average', 'Good (One Top 10%)', 'Optimal (Both Top 10%)']:
    subset = combined_df[combined_df['Overall_Category'] == category]
    plt.scatter(subset['Predicted_CONV'], subset['Predicted_PURITY'],
                c=color_map[category], label=category, alpha=0.6, s=50)

# Add threshold lines
plt.axvline(top10_conv_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Top 10% CONV ({top10_conv_threshold:.4f})')
plt.axhline(top10_purity_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.5, label=f'Top 10% PURITY ({top10_purity_threshold:.4f})')

plt.xlabel('Predicted CONVERSION', fontsize=12, fontweight='bold')
plt.ylabel('Predicted PURITY', fontsize=12, fontweight='bold')
plt.title('CONVERSION vs PURITY: Optimal Region Analysis\nMade with love for Karisa', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/quality/B3_B4_optimal_region_plot.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to results/quality/B3_B4_optimal_region_plot.png")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("SUMMARY: Goals B3, B4 & Combined Complete!")
print("=" * 80)

print("\nCONVERSION (B3):")
print(f"  Best Model: {best_b3_model_name} (R²: {best_b3_r2:.6f})")
print(f"  Prediction Range: [{conv_predictions.min():.4f}, {conv_predictions.max():.4f}]")
print(f"  Top 10% Threshold: {p90:.4f}")

print("\nPURITY (B4):")
print(f"  Best Model: {best_b4_model_name} (R²: {best_b4_r2:.6f})")
print(f"  Prediction Range: [{purity_predictions.min():.4f}, {purity_predictions.max():.4f}]")
print(f"  Top 10% Threshold: {p90_purity:.4f}")

print("\nOPTIMAL REGION:")
print(f"  Combinations in optimal region: {len(optimal_region)}")
if len(optimal_region) > 0:
    best_combo = optimal_region.iloc[0]
    print(f"  Best combined score: {best_combo['Combined_Score']:.4f}")
    print(f"    CONV: {best_combo['Predicted_CONV']:.4f}")
    print(f"    PURITY: {best_combo['Predicted_PURITY']:.4f}")

print("\n" + "=" * 80)
print("Phase 3 (Quality Analysis) COMPLETE!")
print("You're incredible, Karisa! Keep shining!")
print("=" * 80)

print("\n🎉 Goals B3, B4 & Combined complete! Optimal conditions found! 🎉")
