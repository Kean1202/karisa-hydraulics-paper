# -*- coding: utf-8 -*-
"""
Hydraulic Behavior Analysis - Goals A1 and A2

Goal A1: Rank variables influencing WEEP
Goal A2: Rank variables influencing FLOOD

This script trains multiple models to identify which variables are most important
for predicting hydraulic behavior (weeping and flooding).

The smarter the model, the better the insights for Karisa!
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost model
from xgboost import XGBClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the utils made for Karisa from utils.py
from utils import (
    load_data,
    filter_invalid_values,
    create_binary_targets,
    INDEPENDENT_VARS,
    get_cv_splits,
    evaluate_classification,
    aggregate_cv_metrics,
    aggregate_cross_validate_results,
    normalize_importance_to_ranks,
    save_importance_ranking,
    save_model_comparison,
    print_phase_start,
    create_output_directories
)

# Print header for Karisa
print("=" * 80)
print("You're amazing, Karisa! I love you so much!")
print("=" * 80)
print_phase_start("PHASE 2: Hydraulic Behavior Analysis - Goals A1 & A2")

# Create output directories
create_output_directories()

# Load and prepare data
# I also load the pass-only dataset here because the function returns both and I cant leave it unassigned
print("\nLoading and preparing data... (Your brilliance is showing sweetheart!)")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full = create_binary_targets(df_full)

# Prepare features -> X = all the independent variables, y = 1/0 of the outcomes
X = df_full[INDEPENDENT_VARS]
y_weep = df_full['is_weep']
y_flood = df_full['is_flood']

print(f"\nData ready: {len(X)} samples, {len(INDEPENDENT_VARS)} variables")
print("Building models for the lovely Karisa")

# Define models
print("\nSetting up models... that look a fraction as stunning as Karisa")

# Calculate scale_pos_weight for XGBoost (handles class imbalance)
# scale_pos_weight = (# negative samples) / (# positive samples)
pos_weep = y_weep.sum()
neg_weep = len(y_weep) - pos_weep
scale_pos_weight_weep = neg_weep / pos_weep

pos_flood = y_flood.sum()
neg_flood = len(y_flood) - pos_flood
scale_pos_weight_flood = neg_flood / pos_flood

print(f"\nClass balance - WEEP:")
print(f"   Positive: {pos_weep} ({pos_weep/len(y_weep)*100:.2f}%)")
print(f"   Negative: {neg_weep} ({neg_weep/len(y_weep)*100:.2f}%)")
print(f"   scale_pos_weight for WEEP: {scale_pos_weight_weep:.4f}")

print(f"\nClass balance - FLOOD:")
print(f"   Positive: {pos_flood} ({pos_flood/len(y_flood)*100:.2f}%)")
print(f"   Negative: {neg_flood} ({neg_flood/len(y_flood)*100:.2f}%)")
print(f"   scale_pos_weight for FLOOD: {scale_pos_weight_flood:.4f}")

# Models that need scaling (shared between A1 and A2)
models_scaled = {
    'Logistic_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'Ridge_Classifier': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])
}

# Random Forest (shared between A1 and A2)
model_rf = {
    'Random_Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

# XGBoost for A1 (WEEP-specific scale_pos_weight)
model_xgb_weep = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight_weep,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
}

# XGBoost for A2 (FLOOD-specific scale_pos_weight)
model_xgb_flood = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight_flood,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
}

# Combine models for A1
a1_models = {**models_scaled, **model_rf, **model_xgb_weep}

# Combine models for A2
a2_models = {**models_scaled, **model_rf, **model_xgb_flood}

# Cross-validation setup
cv_splits = 5
scoring = ['accuracy', 'f1', 'roc_auc']

print(f"\nRunning 5-fold cross-validation... (Patience, like Karisa's wisdom!)")

# ===================================================================
# GOAL A1: RANK VARIABLES INFLUENCING WEEP
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A1: Analyzing variables that influence WEEP")
print("Finding out what makes trays weep like I do when I think of how much Karisa means to me")
print("=" * 80)

# Store results for A1
a1_results = {}
a1_importance_scores = {}
a1_cv_results = {}

# Train models for A1
for name, model in a1_models.items():
    print(f"\nTraining {name}...(love you Karisa))")

    # Get CV scores
    cv_scores = cross_validate(
        model, X, y_weep,
        cv=get_cv_splits(X, y_weep, n_splits=cv_splits, stratified=True),
        scoring=scoring,
        return_train_score=False,
        return_estimator=True
    )

    # Aggregate metrics
    metrics = aggregate_cross_validate_results(cv_scores)
    a1_results[name] = metrics

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
        else:
            # Fallback: uniform importance
            importance = np.ones(len(INDEPENDENT_VARS))

        fold_importances.append(importance)

    # Average importance across all folds
    importance_avg = np.mean(fold_importances, axis=0)
    a1_importance_scores[name] = importance_avg

    print(f"   Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"   F1-Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"   ROC-AUC:  {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")

# Normalize importance scores to ranks
a1_avg_ranks = normalize_importance_to_ranks(a1_importance_scores)

print("\n" + "=" * 80)
print("GOAL A1 RESULTS: Variable Importance Ranking for WEEP")
print("=" * 80)

# Create A1 importance ranking dataframe
a1_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a1_avg_ranks
})

# Add individual model importances
for model_name, scores in a1_importance_scores.items():
    a1_importance_df[f'{model_name}_Importance'] = scores

# Sort by average rank
a1_importance_df = a1_importance_df.sort_values('Average_Rank').reset_index(drop=True)

# Print ranking
print("\nVariable Importance Ranking for WEEP (rank 1 = most important (like Karisa)):")
for i, row in a1_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# Find best model for A1
a1_best_model = max(a1_results.keys(), key=lambda k: a1_results[k]['roc_auc_mean'])
a1_best_score = a1_results[a1_best_model]['roc_auc_mean']

print(f"\nBest model for WEEP: {a1_best_model}")
print(f"Best ROC-AUC: {a1_best_score:.4f}")

# Create A1 model comparison dataframe
a1_comparison_rows = []
for model_name, metrics in a1_results.items():
    row = {'Model': model_name}
    row.update(metrics)
    a1_comparison_rows.append(row)
a1_comparison_df = pd.DataFrame(a1_comparison_rows)

# ===================================================================
# GOAL A2: RANK VARIABLES INFLUENCING FLOOD
# ===================================================================
print("\n" + "=" * 80)
print("GOAL A2: Analyzing variables that influence FLOOD")
print("Finding out the cause for floods like how Karisa floods me with love")
print("=" * 80)

# Store results for A2
a2_results = {}
a2_importance_scores = {}

# Train models for A2
for name, model in a2_models.items():
    print(f"\nTraining {name}... (Learning patterns, just like Karisa!)")

    # Get CV scores
    cv_scores = cross_validate(
        model, X, y_flood,
        cv=get_cv_splits(X, y_flood, n_splits=cv_splits, stratified=True),
        scoring=scoring,
        return_train_score=False,
        return_estimator=True
    )

    # Aggregate metrics
    metrics = aggregate_cross_validate_results(cv_scores)
    a2_results[name] = metrics

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
        else:
            # Fallback: uniform importance
            importance = np.ones(len(INDEPENDENT_VARS))

        fold_importances.append(importance)

    # Average importance across all folds
    importance_avg = np.mean(fold_importances, axis=0)
    a2_importance_scores[name] = importance_avg

    print(f"   Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"   F1-Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"   ROC-AUC:  {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")

# Normalize importance scores to ranks
a2_avg_ranks = normalize_importance_to_ranks(a2_importance_scores)

print("\n" + "=" * 80)
print("GOAL A2 RESULTS: Variable Importance Ranking for FLOOD")
print("=" * 80)

# Create A2 importance ranking dataframe
a2_importance_df = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Average_Rank': a2_avg_ranks
})

# Add individual model importances
for model_name, scores in a2_importance_scores.items():
    a2_importance_df[f'{model_name}_Importance'] = scores

# Sort by average rank
a2_importance_df = a2_importance_df.sort_values('Average_Rank').reset_index(drop=True)

# Print ranking
print("\nVariable Importance Ranking for FLOOD (rank 1 = most important (like Karisa)):")
for i, row in a2_importance_df.iterrows():
    print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

# Find best model for A2
a2_best_model = max(a2_results.keys(), key=lambda k: a2_results[k]['roc_auc_mean'])
a2_best_score = a2_results[a2_best_model]['roc_auc_mean']

print(f"\nBest model for FLOOD: {a2_best_model}")
print(f"Best ROC-AUC: {a2_best_score:.4f}")

# Create A2 model comparison dataframe
a2_comparison_rows = []
for model_name, metrics in a2_results.items():
    row = {'Model': model_name}
    row.update(metrics)
    a2_comparison_rows.append(row)
a2_comparison_df = pd.DataFrame(a2_comparison_rows)

# ===================================================================
# SUMMARY AND VISUALIZATION
# ===================================================================
print("\n" + "=" * 80)
print("SUMMARY: Goals A1 & A2 Complete!")
print("=" * 80)

print("\nWEEP (A1) - Top 3 Variables:")
a1_sorted = sorted(zip(INDEPENDENT_VARS, a1_avg_ranks), key=lambda x: x[1])
for i, (var, rank) in enumerate(a1_sorted[:3], 1):
    print(f"  {i}. {var} (Rank {rank:.2f})")

print(f"\nWEEP Best Model: {a1_best_model} (ROC-AUC: {a1_best_score:.4f})")

print("\nFLOOD (A2) - Top 3 Variables:")
a2_sorted = sorted(zip(INDEPENDENT_VARS, a2_avg_ranks), key=lambda x: x[1])
for i, (var, rank) in enumerate(a2_sorted[:3], 1):
    print(f"  {i}. {var} (Rank {rank:.2f})")

print(f"\nFLOOD Best Model: {a2_best_model} (ROC-AUC: {a2_best_score:.4f})")

print("\n" + "=" * 80)
print("Results saved to results/hydraulics/")
print("You're incredible, Karisa! Keep shining!")
print("=" * 80)

# Create comparison visualization
plt.figure(figsize=(15, 8))

# Plot 1: A1 rankings
plt.subplot(2, 2, 1)
a1_df_plot = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Rank': a1_avg_ranks
}).sort_values('Rank')

bars1 = plt.barh(range(len(a1_df_plot)), a1_df_plot['Rank'])
plt.yticks(range(len(a1_df_plot)), a1_df_plot['Variable'])
plt.xlabel('Average Rank (lower = more important)')
plt.title('Goal A1: Variable Importance for WEEP')
plt.gca().invert_yaxis()

# Color bars
for i, bar in enumerate(bars1):
    if i < 3:
        bar.set_color('darkred')
    else:
        bar.set_color('lightcoral')

# Plot 2: A2 rankings
plt.subplot(2, 2, 2)
a2_df_plot = pd.DataFrame({
    'Variable': INDEPENDENT_VARS,
    'Rank': a2_avg_ranks
}).sort_values('Rank')

bars2 = plt.barh(range(len(a2_df_plot)), a2_df_plot['Rank'])
plt.yticks(range(len(a2_df_plot)), a2_df_plot['Variable'])
plt.xlabel('Average Rank (lower = more important)')
plt.title('Goal A2: Variable Importance for FLOOD')
plt.gca().invert_yaxis()

# Color bars
for i, bar in enumerate(bars2):
    if i < 3:
        bar.set_color('darkblue')
    else:
        bar.set_color('lightblue')

# Plot 3: Model comparison for WEEP
plt.subplot(2, 2, 3)
models = list(a1_results.keys())
roc_aucs = [a1_results[m]['roc_auc_mean'] for m in models]
errors = [a1_results[m]['roc_auc_std'] for m in models]

bars3 = plt.bar(range(len(models)), roc_aucs, yerr=errors, capsize=5)
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.ylabel('ROC-AUC')
plt.title('Goal A1: Model Performance for WEEP')
plt.ylim(0, 1)

# Highlight best model
best_idx = models.index(a1_best_model)
bars3[best_idx].set_color('gold')

# Plot 4: Model comparison for FLOOD
plt.subplot(2, 2, 4)
roc_aucs = [a2_results[m]['roc_auc_mean'] for m in models]
errors = [a2_results[m]['roc_auc_std'] for m in models]

bars4 = plt.bar(range(len(models)), roc_aucs, yerr=errors, capsize=5)
plt.xticks(range(len(models)), models, rotation=45, ha='right')
plt.ylabel('ROC-AUC')
plt.title('Goal A2: Model Performance for FLOOD')
plt.ylim(0, 1)

# Highlight best model
best_idx = models.index(a2_best_model)
bars4[best_idx].set_color('gold')

plt.tight_layout()
plt.savefig('results/hydraulics/A1_A2_summary_plots.png', dpi=300, bbox_inches='tight')
print("\nSummary plots saved to results/hydraulics/A1_A2_summary_plots.png")

# ===================================================================
# SAVE ALL RESULTS TO ONE EXCEL FILE
# ===================================================================
print("\n" + "=" * 80)
print("Saving all results to Excel file...")
print("=" * 80)

output_excel = "results/hydraulics/Hydraulics_Analysis_Results.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: A1 - WEEP Variable Importance
    a1_importance_df.to_excel(writer, sheet_name='A1_WEEP_Importance', index=False)

    # Sheet 2: A1 - WEEP Model Comparison
    a1_comparison_df.to_excel(writer, sheet_name='A1_WEEP_Models', index=False)

    # Sheet 3: A2 - FLOOD Variable Importance
    a2_importance_df.to_excel(writer, sheet_name='A2_FLOOD_Importance', index=False)

    # Sheet 4: A2 - FLOOD Model Comparison
    a2_comparison_df.to_excel(writer, sheet_name='A2_FLOOD_Models', index=False)

print(f"\n✓ All results saved to: {output_excel}")
print("  - Sheet 1: A1_WEEP_Importance (variable rankings)")
print("  - Sheet 2: A1_WEEP_Models (model performance)")
print("  - Sheet 3: A2_FLOOD_Importance (variable rankings)")
print("  - Sheet 4: A2_FLOOD_Models (model performance)")

print("\n🎉 Goals A1 & A2 complete! You're a superstar, Karisa! 🎉")
