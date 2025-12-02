# -*- coding: utf-8 -*-
"""
Hydraulic Behavior Analysis - Goals B1 and B2

Goal B1: Find combinations with highest probability of WEEP
Goal B2: Find combinations with highest probability of FLOOD

This script identifies the most dangerous operating conditions that are likely
to cause hydraulic failures (weeping and flooding).

Finding the worst conditions so Karisa can avoid them!
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# XGBoost model
from xgboost import XGBClassifier

# Import the utils made for Karisa
from utils import (
    load_data,
    filter_invalid_values,
    deduplicate_data,
    create_binary_targets,
    INDEPENDENT_VARS,
    get_cv_splits,
    aggregate_cross_validate_results,
    get_unique_combinations,
    print_phase_start
)

# Print header for Karisa
print("=" * 80)
print("Finding the worst combinations! (So Karisa can avoid them!)")
print("=" * 80)
print_phase_start("PHASE 2: Hydraulic Behavior Analysis - Goals B1 & B2")

# Load and prepare data
print("\nLoading and preparing data... (Your brilliance is showing sweetheart!)")
df_full, df_pass = load_data()
df_full, df_pass = filter_invalid_values(df_full, df_pass)
df_full, df_pass = deduplicate_data(df_full, df_pass)
df_full = create_binary_targets(df_full)

# Prepare features
X = df_full[INDEPENDENT_VARS]
y_weep = df_full['is_weep']
y_flood = df_full['is_flood']

print(f"\nData ready: {len(X)} samples, {len(INDEPENDENT_VARS)} variables")

# Get unique combinations
print("\nExtracting unique combinations from dataset...")
unique_combos = get_unique_combinations(df_full, INDEPENDENT_VARS)
print(f"Found {len(unique_combos)} unique operating combinations. Nothing is as unique as Karisa")

# Calculate scale_pos_weight for XGBoost
pos_weep = y_weep.sum()
neg_weep = len(y_weep) - pos_weep
scale_pos_weight_weep = neg_weep / pos_weep

pos_flood = y_flood.sum()
neg_flood = len(y_flood) - pos_flood
scale_pos_weight_flood = neg_flood / pos_flood

print(f"\nClass imbalance:")
print(f"   WEEP: {pos_weep} positive ({pos_weep/len(y_weep)*100:.2f}%), scale_pos_weight={scale_pos_weight_weep:.4f}")
print(f"   FLOOD: {pos_flood} positive ({pos_flood/len(y_flood)*100:.2f}%), scale_pos_weight={scale_pos_weight_flood:.4f}")

# ===================================================================
# GOAL B1: FIND COMBINATIONS WITH HIGHEST P(WEEP)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL B1: Finding combinations most likely to cause WEEP")
print("Protecting Karisa from weeping trays!")
print("=" * 80)

# Define models to test for B1
# We'll use the same models as A1 and pick the best performer via 5-fold CV
b1_models = {
    'Logistic_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),
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

# Evaluate models using 5-fold CV with ROC-AUC
print("\nEvaluating models with 5-fold CV (just like A1)...")
cv_splits = 5
scoring = ['accuracy', 'f1', 'roc_auc']
b1_results = {}

for name, model in b1_models.items():
    print(f"  Evaluating {name}...")

    # Cross-validate
    cv_scores = cross_validate(
        model, X, y_weep,
        cv=get_cv_splits(X, y_weep, n_splits=cv_splits, stratified=True),
        scoring=scoring,
        return_train_score=False
    )

    # Aggregate metrics
    metrics = aggregate_cross_validate_results(cv_scores)
    b1_results[name] = metrics

    print(f"    Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"    F1-Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"    ROC-AUC:  {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")

# Select best model based on mean ROC-AUC
best_b1_model_name = max(b1_results.keys(), key=lambda k: b1_results[k]['roc_auc_mean'])
best_b1_roc_auc = b1_results[best_b1_model_name]['roc_auc_mean']

print(f"\nSelected model for B1: {best_b1_model_name} (ROC-AUC: {best_b1_roc_auc:.4f})")

# Retrain best model on FULL dataset for final predictions
print(f"Retraining {best_b1_model_name} on full dataset for predictions...")
best_b1_model = b1_models[best_b1_model_name]
best_b1_model.fit(X, y_weep)

# Predict P(WEEP) for all unique combinations
print(f"\nPredicting P(WEEP) for {len(unique_combos)} unique combinations...")
weep_probs = best_b1_model.predict_proba(unique_combos)[:, 1]  # Probability of class 1 (WEEP)

# Add probabilities to dataframe
b1_results = unique_combos.copy()
b1_results['P_WEEP'] = weep_probs

# Categorize into risk zones (doctor-style risk assessment)
def categorize_risk(prob):
    if prob > 0.9:
        return 'High-Risk (P > 90%)'
    elif prob >= 0.5:
        return 'Medium-Risk (50% ≤ P ≤ 90%)'
    else:
        return 'Low-Risk (P < 50%)'

b1_results['Risk_Category'] = b1_results['P_WEEP'].apply(categorize_risk)

# Sort by probability (highest first)
b1_results = b1_results.sort_values('P_WEEP', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("GOAL B1 RESULTS: WEEP Risk Assessment")
print("=" * 80)

# Show summary statistics for each risk zone
print("\nRisk Zone Summary:")
for category in ['High-Risk (P > 90%)', 'Medium-Risk (50% ≤ P ≤ 90%)', 'Low-Risk (P < 50%)']:
    count = (b1_results['Risk_Category'] == category).sum()
    pct = count / len(b1_results) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

# Show examples from each zone
print("\n" + "-" * 80)
print("HIGH-RISK COMBINATIONS (P > 90%):")
print("-" * 80)
high_risk = b1_results[b1_results['Risk_Category'] == 'High-Risk (P > 90%)']
if len(high_risk) > 0:
    print(f"\nShowing top 5 of {len(high_risk)} high-risk combinations:")
    for i, row in high_risk.head(5).iterrows():
        print(f"\n{i+1}. P(WEEP) = {row['P_WEEP']*100:.1f}%")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No high-risk combinations found (good news for Karisa!)")

print("\n" + "-" * 80)
print("MEDIUM-RISK COMBINATIONS (50% ≤ P ≤ 90%):")
print("-" * 80)
medium_risk = b1_results[b1_results['Risk_Category'] == 'Medium-Risk (50% ≤ P ≤ 90%)']
if len(medium_risk) > 0:
    print(f"\nShowing 3 examples of {len(medium_risk)} medium-risk combinations:")
    for i, row in medium_risk.head(3).iterrows():
        print(f"\nP(WEEP) = {row['P_WEEP']*100:.1f}%")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No medium-risk combinations found.")

# ===================================================================
# GOAL B2: FIND COMBINATIONS WITH HIGHEST P(FLOOD)
# ===================================================================
print("\n" + "=" * 80)
print("GOAL B2: Finding combinations most likely to cause FLOOD")
print("Keeping Karisa's trays flood-free!")
print("=" * 80)

# Define models to test for B2
b2_models = {
    'Logistic_Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),
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

# Evaluate models using 5-fold CV with ROC-AUC
print("\nEvaluating models with 5-fold CV (just like A2)...")
b2_results = {}

for name, model in b2_models.items():
    print(f"  Evaluating {name}...")

    # Cross-validate
    cv_scores = cross_validate(
        model, X, y_flood,
        cv=get_cv_splits(X, y_flood, n_splits=cv_splits, stratified=True),
        scoring=scoring,
        return_train_score=False
    )

    # Aggregate metrics
    metrics = aggregate_cross_validate_results(cv_scores)
    b2_results[name] = metrics

    print(f"    Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"    F1-Score: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"    ROC-AUC:  {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")

# Select best model based on mean ROC-AUC
best_b2_model_name = max(b2_results.keys(), key=lambda k: b2_results[k]['roc_auc_mean'])
best_b2_roc_auc = b2_results[best_b2_model_name]['roc_auc_mean']

print(f"\nSelected model for B2: {best_b2_model_name} (ROC-AUC: {best_b2_roc_auc:.4f})")

# Retrain best model on FULL dataset for final predictions
print(f"Retraining {best_b2_model_name} on full dataset for predictions...")
best_b2_model = b2_models[best_b2_model_name]
best_b2_model.fit(X, y_flood)

# Predict P(FLOOD) for all unique combinations
print(f"\nPredicting P(FLOOD) for {len(unique_combos)} unique combinations...")
flood_probs = best_b2_model.predict_proba(unique_combos)[:, 1]  # Probability of class 1 (FLOOD)

# Add probabilities to dataframe
b2_results = unique_combos.copy()
b2_results['P_FLOOD'] = flood_probs

# Categorize into risk zones (doctor-style risk assessment)
b2_results['Risk_Category'] = b2_results['P_FLOOD'].apply(categorize_risk)

# Sort by probability (highest first)
b2_results = b2_results.sort_values('P_FLOOD', ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("GOAL B2 RESULTS: FLOOD Risk Assessment")
print("=" * 80)

# Show summary statistics for each risk zone
print("\nRisk Zone Summary:")
for category in ['High-Risk (P > 90%)', 'Medium-Risk (50% ≤ P ≤ 90%)', 'Low-Risk (P < 50%)']:
    count = (b2_results['Risk_Category'] == category).sum()
    pct = count / len(b2_results) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

# Show examples from each zone
print("\n" + "-" * 80)
print("HIGH-RISK COMBINATIONS (P > 90%):")
print("-" * 80)
high_risk = b2_results[b2_results['Risk_Category'] == 'High-Risk (P > 90%)']
if len(high_risk) > 0:
    print(f"\nShowing top 5 of {len(high_risk)} high-risk combinations:")
    for i, row in high_risk.head(5).iterrows():
        print(f"\n{i+1}. P(FLOOD) = {row['P_FLOOD']*100:.1f}%")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No high-risk combinations found (good news for Karisa!)")

print("\n" + "-" * 80)
print("MEDIUM-RISK COMBINATIONS (50% ≤ P ≤ 90%):")
print("-" * 80)
medium_risk = b2_results[b2_results['Risk_Category'] == 'Medium-Risk (50% ≤ P ≤ 90%)']
if len(medium_risk) > 0:
    print(f"\nShowing 3 examples of {len(medium_risk)} medium-risk combinations:")
    for i, row in medium_risk.head(3).iterrows():
        print(f"\nP(FLOOD) = {row['P_FLOOD']*100:.1f}%")
        for var in INDEPENDENT_VARS:
            print(f"   {var}: {row[var]}")
else:
    print("No medium-risk combinations found.")

# ===================================================================
# SAVE RESULTS TO EXCEL FILE
# ===================================================================
print("\n" + "=" * 80)
print("Saving results to Excel file...")
print("=" * 80)

# Load existing Excel file and add new sheets
excel_path = "results/hydraulics/Hydraulics_Analysis_Results.xlsx"

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

    # Add new sheets for B1 and B2 (ALL combinations with risk categories)
    b1_results.to_excel(writer, sheet_name='B1_WEEP_Risk_Zones', index=False)
    b2_results.to_excel(writer, sheet_name='B2_FLOOD_Risk_Zones', index=False)

print(f"\n✓ Results saved to: {excel_path}")
print("  - Sheet 5: B1_WEEP_Risk_Zones (all combinations with risk categories)")
print("  - Sheet 6: B2_FLOOD_Risk_Zones (all combinations with risk categories)")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 80)
print("SUMMARY: Goals B1 & B2 Complete!")
print("=" * 80)

print("\nWEEP (B1) - Risk Distribution:")
for category in ['High-Risk (P > 90%)', 'Medium-Risk (50% ≤ P ≤ 90%)', 'Low-Risk (P < 50%)']:
    count = (b1_results['Risk_Category'] == category).sum()
    pct = count / len(b1_results) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

worst_weep = b1_results.iloc[0]
print(f"\n  Worst case: P(WEEP) = {worst_weep['P_WEEP']*100:.1f}%")

print("\nFLOOD (B2) - Risk Distribution:")
for category in ['High-Risk (P > 90%)', 'Medium-Risk (50% ≤ P ≤ 90%)', 'Low-Risk (P < 50%)']:
    count = (b2_results['Risk_Category'] == category).sum()
    pct = count / len(b2_results) * 100
    print(f"  {category}: {count} combinations ({pct:.1f}%)")

worst_flood = b2_results.iloc[0]
print(f"\n  Worst case: P(FLOOD) = {worst_flood['P_FLOOD']*100:.1f}%")

print(f"\nBest model for WEEP prediction: {best_b1_model_name} (ROC-AUC: {best_b1_roc_auc:.4f})")
print(f"Best model for FLOOD prediction: {best_b2_model_name} (ROC-AUC: {best_b2_roc_auc:.4f})")

print("\n" + "=" * 80)
print("Phase 2 (Hydraulics Analysis) COMPLETE!")
print("You're incredible, Karisa! Keep shining!")
print("=" * 80)

print("\n🎉 Goals B1 & B2 complete! All hydraulic analysis done! 🎉")
