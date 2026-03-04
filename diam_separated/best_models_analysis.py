# -*- coding: utf-8 -*-
"""
Best Models Analysis - DIAM-Separated Version

Same as root best_models_analysis.py but:
- Dataset is filtered to a single DIAM value (set via KARISA_DIAM_FILTER env var)
- DIAM is excluded from the 6 independent variables
- Outputs go to results/diam_separated/DIAM_{value}/

Run via main_diam_separated.py (do not run directly).
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

from utils import (
    load_data,
    filter_invalid_values,
    filter_by_diam,
    create_binary_targets,
    INDEPENDENT_VARS,
    USE_HPA,
    DIAM_FILTER_STR,
    get_cv_splits,
    normalize_importance_to_ranks,
    print_phase_start,
    convert_to_percentage,
    VARIABLE_LABELS,
)

# ===================================================================
# Setup
# ===================================================================
diam_val = float(DIAM_FILTER_STR)
DIAM_LABEL = f"DIAM_{diam_val:g}"
DIAM_OUT = Path(f"results/diam_separated/{DIAM_LABEL}")

print("=" * 80)
print(f"BEST MODELS ANALYSIS - DIAM-Separated  [{DIAM_LABEL}]")
print(f"Variables: {INDEPENDENT_VARS}")
print("=" * 80)
print_phase_start(f"Re-analyzing with Best Models: RF & XGB  [{DIAM_LABEL}]")

plt.rcParams['font.family'] = 'Arial'

# Load, filter, and reduce to this DIAM slice
print("\nLoading and preparing data...")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = filter_by_diam(df_full, df_pass)
df_full = create_binary_targets(df_full)
df_pass = convert_to_percentage(df_pass, columns=['CONV', 'PURITY'])

X_full = df_full[INDEPENDENT_VARS]
y_weep = df_full['is_weep']
y_flood = df_full['is_flood']

X_pass = df_pass[INDEPENDENT_VARS]
y_conv = df_pass['CONV']
y_purity = df_pass['PURITY']

print(f"\nFull dataset: {len(X_full)} samples")
print(f"Pass dataset: {len(X_pass)} samples")

pos_weep = y_weep.sum()
neg_weep = len(y_weep) - pos_weep
scale_pos_weight_weep = neg_weep / pos_weep

pos_flood = y_flood.sum()
neg_flood = len(y_flood) - pos_flood
scale_pos_weight_flood = neg_flood / pos_flood

print(f"\nClass imbalance (for XGBoost):")
print(f"   WEEP:  scale_pos_weight = {scale_pos_weight_weep:.4f}")
print(f"   FLOOD: scale_pos_weight = {scale_pos_weight_flood:.4f}")

cv_splits = 5
magma_cmap = plt.cm.get_cmap('magma')


# ===================================================================
# Helper: run one importance analysis
# ===================================================================
def run_importance(models, X, y, stratified, task_label):
    results = {}
    importance_scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if stratified:
            scoring = ['accuracy', 'f1', 'roc_auc']
        else:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        cv_scores = cross_validate(
            model, X, y,
            cv=get_cv_splits(X, y, n_splits=cv_splits, stratified=stratified),
            scoring=scoring,
            return_train_score=False,
            return_estimator=True
        )

        if stratified:
            results[name] = {
                'accuracy_mean': np.mean(cv_scores['test_accuracy']),
                'accuracy_std': np.std(cv_scores['test_accuracy']),
                'f1_mean': np.mean(cv_scores['test_f1']),
                'f1_std': np.std(cv_scores['test_f1']),
                'roc_auc_mean': np.mean(cv_scores['test_roc_auc']),
                'roc_auc_std': np.std(cv_scores['test_roc_auc'])
            }
            print(f"   Accuracy: {results[name]['accuracy_mean']:.4f} ± {results[name]['accuracy_std']:.4f}")
            print(f"   F1-Score: {results[name]['f1_mean']:.4f} ± {results[name]['f1_std']:.4f}")
            print(f"   ROC-AUC:  {results[name]['roc_auc_mean']:.4f} ± {results[name]['roc_auc_std']:.4f}")
        else:
            rmse = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
            mae = -cv_scores['test_neg_mean_absolute_error']
            r2 = cv_scores['test_r2']
            results[name] = {
                'rmse_mean': np.mean(rmse), 'rmse_std': np.std(rmse),
                'mae_mean': np.mean(mae),   'mae_std': np.std(mae),
                'r2_mean': np.mean(r2),     'r2_std': np.std(r2)
            }
            print(f"   RMSE: {results[name]['rmse_mean']:.6f} ± {results[name]['rmse_std']:.6f}")
            print(f"   MAE:  {results[name]['mae_mean']:.6f} ± {results[name]['mae_std']:.6f}")
            print(f"   R²:   {results[name]['r2_mean']:.6f} ± {results[name]['r2_std']:.6f}")

        fold_importances = [est.feature_importances_ for est in cv_scores['estimator']]
        importance_scores[name] = np.mean(fold_importances, axis=0)

    avg_ranks = normalize_importance_to_ranks(importance_scores)
    importance_df = pd.DataFrame({
        'Variable': INDEPENDENT_VARS,
        'Average_Rank': avg_ranks,
        'RF_Importance': importance_scores['Random_Forest'],
        'XGB_Importance': importance_scores['XGBoost']
    }).sort_values('Average_Rank').reset_index(drop=True)

    print(f"\n{task_label} Results:")
    for i, row in importance_df.iterrows():
        print(f"  {i+1}. {row['Variable']}: Rank {row['Average_Rank']:.2f}")

    return results, importance_scores, importance_df


# ===================================================================
# A1: WEEP
# ===================================================================
print("\n" + "=" * 80)
print(f"A1: WEEP Variable Importance  [{DIAM_LABEL}]")
print("=" * 80)
a1_results, a1_importance_scores, a1_importance_df = run_importance(
    {
        'Random_Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight_weep, random_state=42, n_jobs=-1, eval_metric='logloss')
    },
    X_full, y_weep, stratified=True, task_label="A1 WEEP"
)

# ===================================================================
# A2: FLOOD
# ===================================================================
print("\n" + "=" * 80)
print(f"A2: FLOOD Variable Importance  [{DIAM_LABEL}]")
print("=" * 80)
a2_results, a2_importance_scores, a2_importance_df = run_importance(
    {
        'Random_Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight_flood, random_state=42, n_jobs=-1, eval_metric='logloss')
    },
    X_full, y_flood, stratified=True, task_label="A2 FLOOD"
)

# ===================================================================
# A3: CONVERSION
# ===================================================================
print("\n" + "=" * 80)
print(f"A3: CONVERSION Variable Importance  [{DIAM_LABEL}]")
print("=" * 80)
a3_results, a3_importance_scores, a3_importance_df = run_importance(
    {
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    },
    X_pass, y_conv, stratified=False, task_label="A3 CONV"
)

# ===================================================================
# A4: PURITY
# ===================================================================
print("\n" + "=" * 80)
print(f"A4: PURITY Variable Importance  [{DIAM_LABEL}]")
print("=" * 80)
a4_results, a4_importance_scores, a4_importance_df = run_importance(
    {
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    },
    X_pass, y_purity, stratified=False, task_label="A4 PURITY"
)

# ===================================================================
# Save to Excel
# ===================================================================
print("\n" + "=" * 80)
print("Saving results to Excel...")
DIAM_OUT.mkdir(parents=True, exist_ok=True)
output_excel = DIAM_OUT / "Best_Models_Analysis_Results_NO_HPA.xlsx"

a1_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a1_results.items()])
a2_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a2_results.items()])
a3_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a3_results.items()])
a4_comparison_df = pd.DataFrame([{'Model': k, **v} for k, v in a4_results.items()])

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    a1_importance_df.to_excel(writer, sheet_name='A1_WEEP_Importance', index=False)
    a1_comparison_df.to_excel(writer, sheet_name='A1_WEEP_Models', index=False)
    a2_importance_df.to_excel(writer, sheet_name='A2_FLOOD_Importance', index=False)
    a2_comparison_df.to_excel(writer, sheet_name='A2_FLOOD_Models', index=False)
    a3_importance_df.to_excel(writer, sheet_name='A3_CONV_Importance', index=False)
    a3_comparison_df.to_excel(writer, sheet_name='A3_CONV_Models', index=False)
    a4_importance_df.to_excel(writer, sheet_name='A4_PURITY_Importance', index=False)
    a4_comparison_df.to_excel(writer, sheet_name='A4_PURITY_Models', index=False)

print(f"\n✓ Results saved to: {output_excel}")

# ===================================================================
# Visualizations
# ===================================================================
print("\n" + "=" * 80)
print("Creating visualizations...")
plot_dir = DIAM_OUT / "best_model_no_hpa"
plot_dir.mkdir(parents=True, exist_ok=True)


def save_importance_bar(importance_df, filename, title_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    variables = [VARIABLE_LABELS.get(v, v) for v in importance_df['Variable']]
    rf_scores = importance_df['RF_Importance'].values
    xgb_scores = importance_df['XGB_Importance'].values
    y_pos = np.arange(len(variables))
    bar_height = 0.35
    ax.barh(y_pos - bar_height / 2, rf_scores, bar_height, label='Random Forest', color=magma_cmap(0.3))
    ax.barh(y_pos + bar_height / 2, xgb_scores, bar_height, label='XGBoost', color=magma_cmap(0.7))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Importance Score', fontsize=16, fontfamily='Arial')
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    ax.legend(fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


save_importance_bar(a1_importance_df, 'A1_WEEP_Importance.png', 'A1 WEEP')
save_importance_bar(a2_importance_df, 'A2_FLOOD_Importance.png', 'A2 FLOOD')
save_importance_bar(a3_importance_df, 'A3_CONV_Importance.png', 'A3 CONV')
save_importance_bar(a4_importance_df, 'A4_PURITY_Importance.png', 'A4 PURITY')

print(f"\n✓ All outputs saved under: {DIAM_OUT.absolute()}")
print(f"\n🎉 Best Models Analysis Complete for {DIAM_LABEL}! 🎉")
