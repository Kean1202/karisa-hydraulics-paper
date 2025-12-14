# -*- coding: utf-8 -*-
"""
Best Models Analysis - Complete Re-analysis

This script re-runs ALL analyses using only the best performing models:
- Random Forest
- XGBoost

Goals covered:
- A1: Variable importance for WEEP
- A2: Variable importance for FLOOD
- A3: Variable importance for CONVERSION
- A4: Variable importance for PURITY

This provides cleaner, more focused results using only the top performers.

Made with love for Karisa - using only the best, just like you!
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate

# XGBoost models
from xgboost import XGBClassifier, XGBRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the utils made for Karisa
from utils import (
    load_data,
    filter_invalid_values,
    deduplicate_data,
    create_binary_targets,
    INDEPENDENT_VARS,
    get_cv_splits,
    normalize_importance_to_ranks,
    print_phase_start,
    convert_to_percentage,
    VARIABLE_LABELS,
    format_axis_for_paper
)

# Print header for Karisa
print("=" * 80)
print("BEST MODELS ANALYSIS - Random Forest & XGBoost Only")
print("Using only the champions, just like Karisa!")
print("=" * 80)
print_phase_start("Re-analyzing with Best Models: RF & XGB")

# Set up plotting style
plt.rcParams['font.family'] = 'Arial'

# Load and prepare data
print("\nLoading and preparing data...")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)
df_full = create_binary_targets(df_full)

# Convert CONV and PURITY to percentages
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

# Prepare features for hydraulics (full dataset)
X_full = df_full[INDEPENDENT_VARS]
y_weep = df_full['is_weep']
y_flood = df_full['is_flood']

# Prepare features for quality (pass only dataset)
X_pass = df_pass[INDEPENDENT_VARS]
y_conv = df_pass['CONV']
y_purity = df_pass['PURITY']

print(f"\nFull dataset: {len(X_full)} samples")
print(f"Pass dataset: {len(X_pass)} samples")

# Calculate scale_pos_weight for XGBoost classifiers
pos_weep = y_weep.sum()
neg_weep = len(y_weep) - pos_weep
scale_pos_weight_weep = neg_weep / pos_weep

pos_flood = y_flood.sum()
neg_flood = len(y_flood) - pos_flood
scale_pos_weight_flood = neg_flood / pos_flood

print(f"\nClass imbalance (for XGBoost):")
print(f"   WEEP: scale_pos_weight = {scale_pos_weight_weep:.4f}")
print(f"   FLOOD: scale_pos_weight = {scale_pos_weight_flood:.4f}")

# Cross-validation setup
cv_splits = 5

# ===================================================================
# GOAL A1: RANK VARIABLES INFLUENCING WEEP (BEST MODELS ONLY)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A1: WEEP Variable Importance (RF & XGB Only)")
print("=" * 80)

a1_models = {
    'Random_Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight_weep,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
}

a1_results = {}
a1_importance_scores = {}

for name, model in a1_models.items():
    print(f"\nTraining {name}...")

    cv_scores = cross_validate(
        model, X_full, y_weep,
        cv=get_cv_splits(X_full, y_weep, n_splits=cv_splits, stratified=True),
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=False,
        return_estimator=True
    )

    # Store metrics
    a1_results[name] = {
        'accuracy_mean': np.mean(cv_scores['test_accuracy']),
        'accuracy_std': np.std(cv_scores['test_accuracy']),
        'f1_mean': np.mean(cv_scores['test_f1']),
        'f1_std': np.std(cv_scores['test_f1']),
        'roc_auc_mean': np.mean(cv_scores['test_roc_auc']),
        'roc_auc_std': np.std(cv_scores['test_roc_auc'])
    }

    # Extract importance from all folds and average
    fold_importances = []
    for est in cv_scores['estimator']:
        fold_importances.append(est.feature_importances_)

    importance_avg = np.mean(fold_importances, axis=0)
    a1_importance_scores[name] = importance_avg

    print(f"   Accuracy: {a1_results[name]['accuracy_mean']:.4f} ± {a1_results[name]['accuracy_std']:.4f}")
    print(f"   F1-Score: {a1_results[name]['f1_mean']:.4f} ± {a1_results[name]['f1_std']:.4f}")
    print(f"   ROC-AUC:  {a1_results[name]['roc_auc_mean']:.4f} ± {a1_results[name]['roc_auc_std']:.4f}")

# Normalize importance to ranks
a1_avg_ranks = normalize_importance_to_ranks(a1_importance_scores)

# Create importance dataframe
a1_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a1_avg_ranks,
    'RF_Importance': a1_importance_scores['Random_Forest'],
    'XGB_Importance': a1_importance_scores['XGBoost']
}).sort_values('Average_Rank').reset_index(drop=True)

print("\n" + "=" * 80)
print("A1 RESULTS: WEEP Variable Importance (Best Models)")
print("=" * 80)
for i, row in a1_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# ===================================================================
# GOAL A2: RANK VARIABLES INFLUENCING FLOOD (BEST MODELS ONLY)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A2: FLOOD Variable Importance (RF & XGB Only)")
print("=" * 80)

a2_models = {
    'Random_Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight_flood,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
}

a2_results = {}
a2_importance_scores = {}

for name, model in a2_models.items():
    print(f"\nTraining {name}...")

    cv_scores = cross_validate(
        model, X_full, y_flood,
        cv=get_cv_splits(X_full, y_flood, n_splits=cv_splits, stratified=True),
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=False,
        return_estimator=True
    )

    # Store metrics
    a2_results[name] = {
        'accuracy_mean': np.mean(cv_scores['test_accuracy']),
        'accuracy_std': np.std(cv_scores['test_accuracy']),
        'f1_mean': np.mean(cv_scores['test_f1']),
        'f1_std': np.std(cv_scores['test_f1']),
        'roc_auc_mean': np.mean(cv_scores['test_roc_auc']),
        'roc_auc_std': np.std(cv_scores['test_roc_auc'])
    }

    # Extract importance from all folds and average
    fold_importances = []
    for est in cv_scores['estimator']:
        fold_importances.append(est.feature_importances_)

    importance_avg = np.mean(fold_importances, axis=0)
    a2_importance_scores[name] = importance_avg

    print(f"   Accuracy: {a2_results[name]['accuracy_mean']:.4f} ± {a2_results[name]['accuracy_std']:.4f}")
    print(f"   F1-Score: {a2_results[name]['f1_mean']:.4f} ± {a2_results[name]['f1_std']:.4f}")
    print(f"   ROC-AUC:  {a2_results[name]['roc_auc_mean']:.4f} ± {a2_results[name]['roc_auc_std']:.4f}")

# Normalize importance to ranks
a2_avg_ranks = normalize_importance_to_ranks(a2_importance_scores)

# Create importance dataframe
a2_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a2_avg_ranks,
    'RF_Importance': a2_importance_scores['Random_Forest'],
    'XGB_Importance': a2_importance_scores['XGBoost']
}).sort_values('Average_Rank').reset_index(drop=True)

print("\n" + "=" * 80)
print("A2 RESULTS: FLOOD Variable Importance (Best Models)")
print("=" * 80)
for i, row in a2_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# ===================================================================
# GOAL A3: RANK VARIABLES INFLUENCING CONVERSION (BEST MODELS ONLY)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A3: CONVERSION Variable Importance (RF & XGB Only)")
print("=" * 80)

a3_models = {
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

a3_results = {}
a3_importance_scores = {}

for name, model in a3_models.items():
    print(f"\nTraining {name}...")

    cv_scores = cross_validate(
        model, X_pass, y_conv,
        cv=get_cv_splits(X_pass, y_conv, n_splits=cv_splits, stratified=False),
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False,
        return_estimator=True
    )

    # Compute metrics properly
    rmse_per_fold = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
    mae_per_fold = -cv_scores['test_neg_mean_absolute_error']
    r2_per_fold = cv_scores['test_r2']

    a3_results[name] = {
        'rmse_mean': np.mean(rmse_per_fold),
        'rmse_std': np.std(rmse_per_fold),
        'mae_mean': np.mean(mae_per_fold),
        'mae_std': np.std(mae_per_fold),
        'r2_mean': np.mean(r2_per_fold),
        'r2_std': np.std(r2_per_fold)
    }

    # Extract importance from all folds and average
    fold_importances = []
    for est in cv_scores['estimator']:
        fold_importances.append(est.feature_importances_)

    importance_avg = np.mean(fold_importances, axis=0)
    a3_importance_scores[name] = importance_avg

    print(f"   RMSE: {a3_results[name]['rmse_mean']:.6f} ± {a3_results[name]['rmse_std']:.6f}")
    print(f"   MAE:  {a3_results[name]['mae_mean']:.6f} ± {a3_results[name]['mae_std']:.6f}")
    print(f"   R²:   {a3_results[name]['r2_mean']:.6f} ± {a3_results[name]['r2_std']:.6f}")

# Normalize importance to ranks
a3_avg_ranks = normalize_importance_to_ranks(a3_importance_scores)

# Create importance dataframe
a3_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a3_avg_ranks,
    'RF_Importance': a3_importance_scores['Random_Forest'],
    'XGB_Importance': a3_importance_scores['XGBoost']
}).sort_values('Average_Rank').reset_index(drop=True)

print("\n" + "=" * 80)
print("A3 RESULTS: CONVERSION Variable Importance (Best Models)")
print("=" * 80)
for i, row in a3_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# ===================================================================
# GOAL A4: RANK VARIABLES INFLUENCING PURITY (BEST MODELS ONLY)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A4: PURITY Variable Importance (RF & XGB Only)")
print("=" * 80)

a4_models = {
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

a4_results = {}
a4_importance_scores = {}

for name, model in a4_models.items():
    print(f"\nTraining {name}...")

    cv_scores = cross_validate(
        model, X_pass, y_purity,
        cv=get_cv_splits(X_pass, y_purity, n_splits=cv_splits, stratified=False),
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False,
        return_estimator=True
    )

    # Compute metrics properly
    rmse_per_fold = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
    mae_per_fold = -cv_scores['test_neg_mean_absolute_error']
    r2_per_fold = cv_scores['test_r2']

    a4_results[name] = {
        'rmse_mean': np.mean(rmse_per_fold),
        'rmse_std': np.std(rmse_per_fold),
        'mae_mean': np.mean(mae_per_fold),
        'mae_std': np.std(mae_per_fold),
        'r2_mean': np.mean(r2_per_fold),
        'r2_std': np.std(r2_per_fold)
    }

    # Extract importance from all folds and average
    fold_importances = []
    for est in cv_scores['estimator']:
        fold_importances.append(est.feature_importances_)

    importance_avg = np.mean(fold_importances, axis=0)
    a4_importance_scores[name] = importance_avg

    print(f"   RMSE: {a4_results[name]['rmse_mean']:.6f} ± {a4_results[name]['rmse_std']:.6f}")
    print(f"   MAE:  {a4_results[name]['mae_mean']:.6f} ± {a4_results[name]['mae_std']:.6f}")
    print(f"   R²:   {a4_results[name]['r2_mean']:.6f} ± {a4_results[name]['r2_std']:.6f}")

# Normalize importance to ranks
a4_avg_ranks = normalize_importance_to_ranks(a4_importance_scores)

# Create importance dataframe
a4_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a4_avg_ranks,
    'RF_Importance': a4_importance_scores['Random_Forest'],
    'XGB_Importance': a4_importance_scores['XGBoost']
}).sort_values('Average_Rank').reset_index(drop=True)

print("\n" + "=" * 80)
print("A4 RESULTS: PURITY Variable Importance (Best Models)")
print("=" * 80)
for i, row in a4_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# ===================================================================
# SAVE ALL RESULTS TO EXCEL
# ===================================================================
print("\n" + "=" * 80)
print("Saving results to Excel...")
print("=" * 80)

output_excel = "results/Best_Models_Analysis_Results.xlsx"

# Create model comparison dataframes
a1_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a1_results.items()])
a2_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a2_results.items()])
a3_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a3_results.items()])
a4_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a4_results.items()])

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Hydraulics results
    a1_importance_df.to_excel(writer, sheet_name='A1_WEEP_Importance', index=False)
    a1_comparison_df.to_excel(writer, sheet_name='A1_WEEP_Models', index=False)
    a2_importance_df.to_excel(writer, sheet_name='A2_FLOOD_Importance', index=False)
    a2_comparison_df.to_excel(writer, sheet_name='A2_FLOOD_Models', index=False)

    # Quality results
    a3_importance_df.to_excel(writer, sheet_name='A3_CONV_Importance', index=False)
    a3_comparison_df.to_excel(writer, sheet_name='A3_CONV_Models', index=False)
    a4_importance_df.to_excel(writer, sheet_name='A4_PURITY_Importance', index=False)
    a4_comparison_df.to_excel(writer, sheet_name='A4_PURITY_Models', index=False)

print(f"\n✓ Results saved to: {output_excel}")
print("  - 8 sheets total (importance + models for each goal)")

# ===================================================================
# CREATE INDIVIDUAL VISUALIZATIONS
# ===================================================================
print("\n" + "=" * 80)
print("Creating individual visualizations...")
print("=" * 80)

# Create output directory
from pathlib import Path
output_dir = Path("results/best_model")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================================================================
# A1: WEEP IMPORTANCE (RF & XGB separately)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data
variables = [VARIABLE_LABELS.get(v, v) for v in a1_importance_df['Variable']]
rf_scores = a1_importance_df['RF_Importance'].values
xgb_scores = a1_importance_df['XGB_Importance'].values

# Set up bar positions
y_pos = np.arange(len(variables))
bar_height = 0.35

# Create grouped horizontal bars
bars1 = ax.barh(y_pos - bar_height/2, rf_scores, bar_height, label='Random Forest', color='forestgreen')
bars2 = ax.barh(y_pos + bar_height/2, xgb_scores, bar_height, label='XGBoost', color='darkorange')

ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.set_xlabel('Importance Score', fontsize=16, fontfamily='Arial')
ax.tick_params(axis='both', labelsize=14)
ax.invert_yaxis()
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'A1_WEEP_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: A1_WEEP_Importance.png")

# ===================================================================
# A2: FLOOD IMPORTANCE (RF & XGB separately)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data
variables = [VARIABLE_LABELS.get(v, v) for v in a2_importance_df['Variable']]
rf_scores = a2_importance_df['RF_Importance'].values
xgb_scores = a2_importance_df['XGB_Importance'].values

# Set up bar positions
y_pos = np.arange(len(variables))
bar_height = 0.35

# Create grouped horizontal bars
bars1 = ax.barh(y_pos - bar_height/2, rf_scores, bar_height, label='Random Forest', color='forestgreen')
bars2 = ax.barh(y_pos + bar_height/2, xgb_scores, bar_height, label='XGBoost', color='darkorange')

ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.set_xlabel('Importance Score', fontsize=16, fontfamily='Arial')
ax.tick_params(axis='both', labelsize=14)
ax.invert_yaxis()
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'A2_FLOOD_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: A2_FLOOD_Importance.png")

# ===================================================================
# A3: CONVERSION IMPORTANCE (RF & XGB separately)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data
variables = [VARIABLE_LABELS.get(v, v) for v in a3_importance_df['Variable']]
rf_scores = a3_importance_df['RF_Importance'].values
xgb_scores = a3_importance_df['XGB_Importance'].values

# Set up bar positions
y_pos = np.arange(len(variables))
bar_height = 0.35

# Create grouped horizontal bars
bars1 = ax.barh(y_pos - bar_height/2, rf_scores, bar_height, label='Random Forest', color='forestgreen')
bars2 = ax.barh(y_pos + bar_height/2, xgb_scores, bar_height, label='XGBoost', color='darkorange')

ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.set_xlabel('Importance Score', fontsize=16, fontfamily='Arial')
ax.tick_params(axis='both', labelsize=14)
ax.invert_yaxis()
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'A3_CONV_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: A3_CONV_Importance.png")

# ===================================================================
# A4: PURITY IMPORTANCE (RF & XGB separately)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data
variables = [VARIABLE_LABELS.get(v, v) for v in a4_importance_df['Variable']]
rf_scores = a4_importance_df['RF_Importance'].values
xgb_scores = a4_importance_df['XGB_Importance'].values

# Set up bar positions
y_pos = np.arange(len(variables))
bar_height = 0.35

# Create grouped horizontal bars
bars1 = ax.barh(y_pos - bar_height/2, rf_scores, bar_height, label='Random Forest', color='forestgreen')
bars2 = ax.barh(y_pos + bar_height/2, xgb_scores, bar_height, label='XGBoost', color='darkorange')

ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.set_xlabel('Importance Score', fontsize=16, fontfamily='Arial')
ax.tick_params(axis='both', labelsize=14)
ax.invert_yaxis()
ax.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'A4_PURITY_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: A4_PURITY_Importance.png")

print(f"\n✓ All visualizations saved to: {output_dir.absolute()}")

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("COMPLETE SUMMARY - BEST MODELS ANALYSIS")
print("=" * 80)

print("\nA1 - WEEP (Top 3 Variables):")
for i, row in a1_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print("\nA2 - FLOOD (Top 3 Variables):")
for i, row in a2_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print("\nA3 - CONVERSION (Top 3 Variables):")
for i, row in a3_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print("\nA4 - PURITY (Top 3 Variables):")
for i, row in a4_importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Variable']} (Rank {row['Average_Rank']:.2f})")

print("\n" + "=" * 80)
print("Analysis complete using only the best models!")
print("Clean, focused results for Karisa's amazing project!")
print("=" * 80)

print("\n🎉 Best Models Analysis Complete! Champion-level results! 🎉")
