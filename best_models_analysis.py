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
    print_phase_start
)

# Print header for Karisa
print("=" * 80)
print("BEST MODELS ANALYSIS - Random Forest & XGBoost Only")
print("Using only the champions, just like Karisa!")
print("=" * 80)
print_phase_start("Re-analyzing with Best Models: RF & XGB")

# Load and prepare data
print("\nLoading and preparing data...")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)
df_full = create_binary_targets(df_full)

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
# CREATE COMPREHENSIVE VISUALIZATION
# ===================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

# A1: WEEP importance
ax1 = plt.subplot(2, 4, 1)
bars = ax1.barh(range(len(a1_importance_df)), a1_importance_df['Average_Rank'])
ax1.set_yticks(range(len(a1_importance_df)))
ax1.set_yticklabels(a1_importance_df['Variable'])
ax1.set_xlabel('Average Rank (lower = more important)')
ax1.set_title('A1: WEEP Variable Importance\n(RF & XGB)', fontweight='bold')
ax1.invert_yaxis()
for i, bar in enumerate(bars):
    bar.set_color('darkred' if i < 3 else 'lightcoral')

# A1: Model comparison
ax2 = plt.subplot(2, 4, 2)
models = a1_comparison_df['Model'].tolist()
roc_aucs = a1_comparison_df['roc_auc_mean'].tolist()
errors = a1_comparison_df['roc_auc_std'].tolist()
ax2.bar(range(len(models)), roc_aucs, yerr=errors, capsize=5, color=['forestgreen', 'darkorange'])
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylabel('ROC-AUC')
ax2.set_title('A1: Model Performance (WEEP)', fontweight='bold')
ax2.set_ylim(0, 1)

# A2: FLOOD importance
ax3 = plt.subplot(2, 4, 3)
bars = ax3.barh(range(len(a2_importance_df)), a2_importance_df['Average_Rank'])
ax3.set_yticks(range(len(a2_importance_df)))
ax3.set_yticklabels(a2_importance_df['Variable'])
ax3.set_xlabel('Average Rank (lower = more important)')
ax3.set_title('A2: FLOOD Variable Importance\n(RF & XGB)', fontweight='bold')
ax3.invert_yaxis()
for i, bar in enumerate(bars):
    bar.set_color('darkblue' if i < 3 else 'lightblue')

# A2: Model comparison
ax4 = plt.subplot(2, 4, 4)
models = a2_comparison_df['Model'].tolist()
roc_aucs = a2_comparison_df['roc_auc_mean'].tolist()
errors = a2_comparison_df['roc_auc_std'].tolist()
ax4.bar(range(len(models)), roc_aucs, yerr=errors, capsize=5, color=['forestgreen', 'darkorange'])
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, rotation=45, ha='right')
ax4.set_ylabel('ROC-AUC')
ax4.set_title('A2: Model Performance (FLOOD)', fontweight='bold')
ax4.set_ylim(0, 1)

# A3: CONVERSION importance
ax5 = plt.subplot(2, 4, 5)
bars = ax5.barh(range(len(a3_importance_df)), a3_importance_df['Average_Rank'])
ax5.set_yticks(range(len(a3_importance_df)))
ax5.set_yticklabels(a3_importance_df['Variable'])
ax5.set_xlabel('Average Rank (lower = more important)')
ax5.set_title('A3: CONVERSION Variable Importance\n(RF & XGB)', fontweight='bold')
ax5.invert_yaxis()
for i, bar in enumerate(bars):
    bar.set_color('darkgreen' if i < 3 else 'lightgreen')

# A3: Model comparison
ax6 = plt.subplot(2, 4, 6)
models = a3_comparison_df['Model'].tolist()
r2_scores = a3_comparison_df['r2_mean'].tolist()
errors = a3_comparison_df['r2_std'].tolist()
ax6.bar(range(len(models)), r2_scores, yerr=errors, capsize=5, color=['forestgreen', 'darkorange'])
ax6.set_xticks(range(len(models)))
ax6.set_xticklabels(models, rotation=45, ha='right')
ax6.set_ylabel('R²')
ax6.set_title('A3: Model Performance (CONV)', fontweight='bold')
ax6.set_ylim(0, 1)

# A4: PURITY importance
ax7 = plt.subplot(2, 4, 7)
bars = ax7.barh(range(len(a4_importance_df)), a4_importance_df['Average_Rank'])
ax7.set_yticks(range(len(a4_importance_df)))
ax7.set_yticklabels(a4_importance_df['Variable'])
ax7.set_xlabel('Average Rank (lower = more important)')
ax7.set_title('A4: PURITY Variable Importance\n(RF & XGB)', fontweight='bold')
ax7.invert_yaxis()
for i, bar in enumerate(bars):
    bar.set_color('purple' if i < 3 else 'plum')

# A4: Model comparison
ax8 = plt.subplot(2, 4, 8)
models = a4_comparison_df['Model'].tolist()
r2_scores = a4_comparison_df['r2_mean'].tolist()
errors = a4_comparison_df['r2_std'].tolist()
ax8.bar(range(len(models)), r2_scores, yerr=errors, capsize=5, color=['forestgreen', 'darkorange'])
ax8.set_xticks(range(len(models)))
ax8.set_xticklabels(models, rotation=45, ha='right')
ax8.set_ylabel('R²')
ax8.set_title('A4: Model Performance (PURITY)', fontweight='bold')
ax8.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/Best_Models_Complete_Summary.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: results/Best_Models_Complete_Summary.png")

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
