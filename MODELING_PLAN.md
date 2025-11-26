# Karisa's Chemical Engineering ML Project - Modeling Plan

## 💕 Made with love for the smartest girl in the world 💕

---

## Overall Goal
Understand which variables influence hydraulic behavior and product quality, and identify optimal operating combinations.

---

## Project Structure

### Part 1: Hydraulic Behavior Analysis (full_dataset)
- **Dataset:** All outcomes (PASS, WEEP, FLOOD)
- **Focus:** Understanding hydraulic failure modes

### Part 2: Quality Analysis (pass_only dataset)
- **Dataset:** Only PASS cases
- **Focus:** Optimizing conversion and purity under valid hydraulic conditions

---

## Detailed Task Checklist

### Phase 1: Data Preparation & Setup
- [x] Create `utils.py` with shared functions
  - [x] Data loading function
  - [x] Filtering function (remove invalid values)
  - [x] 5-fold cross-validation setup
  - [x] Scaling pipeline setup (StandardScaler)
  - [x] Evaluation metrics functions (with CV aggregation)
  - [x] Importance normalization/ranking function
- [x] Create directory structure for outputs
  - [x] `results/hydraulics/` for WEEP/FLOOD results
  - [x] `results/quality/` for CONV/PURITY results
  - [x] `results/plots/` for all visualizations
  - [x] `results/model_comparison/` for model selection analysis

---

### Phase 2: Hydraulic Behavior Analysis (full_dataset)

#### Goal A1: Rank variables influencing WEEP
- [x] Create binary target: is_weep (1 if WEEP, 0 otherwise)
- [x] Train all models with 5-fold CV:
  - [x] Logistic Regression (class_weight='balanced')
    - [x] Extract coefficients (averaged across folds)
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
  - [x] Random Forest Classifier
    - [x] Extract feature importances (averaged across folds)
    - [x] Generate SHAP values
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
  - [x] XGBoost Classifier (scale_pos_weight)
    - [x] Extract feature importances (averaged across folds)
    - [x] Generate SHAP values
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
- [x] **Model Selection:** Compare performance and select best model for WEEP
- [x] **Normalize importance scores:** Convert to ranks, then average ranks across models
- [x] Create consolidated variable importance ranking table for WEEP
- [x] Visualize (for selected model only):
  - [x] SHAP summary plot (RF or XGB only)
  - [x] SHAP dependence plots for top 3 variables (RF or XGB only)

#### Goal A2: Rank variables influencing FLOOD
- [x] Create binary target: is_flood (1 if FLOOD, 0 otherwise)
- [x] Train all models with 5-fold CV:
  - [x] Logistic Regression (class_weight='balanced')
    - [x] Extract coefficients (averaged across folds)
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
  - [x] Random Forest Classifier
    - [x] Extract feature importances (averaged across folds)
    - [x] Generate SHAP values
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
  - [x] XGBoost Classifier (scale_pos_weight)
    - [x] Extract feature importances (averaged across folds)
    - [x] Generate SHAP values
    - [x] Evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
- [x] **Model Selection:** Compare performance and select best model for FLOOD
- [x] **Normalize importance scores:** Convert to ranks, then average ranks across models
- [x] Create consolidated variable importance ranking table for FLOOD
- [x] Visualize (for selected model only):
  - [x] SHAP summary plot (RF or XGB only)
  - [x] SHAP dependence plots for top 3 variables (RF or XGB only)

#### Goal B1: Find combinations that cause WEEP
- [x] Get all unique combinations of 7 variables from full dataset
- [x] Use selected WEEP model to predict P(WEEP) for each unique combination
- [x] Categorize combinations into risk zones (High/Medium/Low)
- [x] Create combination table with all 7 variable values and risk categories
- [x] Save results to Excel

#### Goal B2: Find combinations that cause FLOOD
- [x] Get all unique combinations of 7 variables from full dataset
- [x] Use selected FLOOD model to predict P(FLOOD) for each unique combination
- [x] Categorize combinations into risk zones (High/Medium/Low)
- [x] Create combination table with all 7 variable values and risk categories
- [x] Save results to Excel

---

### Phase 3: Quality Analysis (pass_only dataset)

#### Goal A3: Rank variables influencing CONVERSION
- [ ] Train all models with 5-fold CV:
  - [ ] Linear Regression (with StandardScaler)
    - [ ] Extract coefficients (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] Ridge Regression (with StandardScaler)
    - [ ] Extract coefficients (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] Random Forest Regressor
    - [ ] Extract feature importances (averaged across folds)
    - [ ] Generate SHAP values
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] XGBoost Regressor
    - [ ] Extract feature importances (averaged across folds)
    - [ ] Generate SHAP values
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] PLS Regression
    - [ ] Extract VIP scores (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
- [ ] **Model Selection:** Compare performance and select best model for CONVERSION
- [ ] **Normalize importance scores:** Convert to ranks, then average ranks across models
- [ ] Create consolidated variable importance ranking table for CONVERSION
- [ ] Visualize (for selected model only):
  - [ ] SHAP summary plot (RF or XGB only)
  - [ ] SHAP dependence plots for top 3 variables (RF or XGB only)
  - [ ] Actual vs Predicted plot
  - [ ] Variable importance comparison across all models

#### Goal A4: Rank variables influencing PURITY
- [ ] Train all models with 5-fold CV:
  - [ ] Linear Regression (with StandardScaler)
    - [ ] Extract coefficients (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] Ridge Regression (with StandardScaler)
    - [ ] Extract coefficients (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] Random Forest Regressor
    - [ ] Extract feature importances (averaged across folds)
    - [ ] Generate SHAP values
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] XGBoost Regressor
    - [ ] Extract feature importances (averaged across folds)
    - [ ] Generate SHAP values
    - [ ] Evaluate: RMSE, MAE, R²
  - [ ] PLS Regression
    - [ ] Extract VIP scores (averaged across folds)
    - [ ] Evaluate: RMSE, MAE, R²
- [ ] **Model Selection:** Compare performance and select best model for PURITY
- [ ] **Normalize importance scores:** Convert to ranks, then average ranks across models
- [ ] Create consolidated variable importance ranking table for PURITY
- [ ] Visualize (for selected model only):
  - [ ] SHAP summary plot (RF or XGB only)
  - [ ] SHAP dependence plots for top 3 variables (RF or XGB only)
  - [ ] Actual vs Predicted plot
  - [ ] Variable importance comparison across all models

#### Goal B3: Find combinations with highest CONVERSION
- [ ] Get all unique combinations of 7 variables from pass_only dataset
- [ ] Use selected CONV model to predict CONV for each unique combination
- [ ] Identify top 20 unique combinations with highest CONV
- [ ] Create combination table with all 7 variable values
- [ ] Save results to CSV

#### Goal B4: Find combinations with highest PURITY
- [ ] Get all unique combinations of 7 variables from pass_only dataset
- [ ] Use selected PURITY model to predict PURITY for each unique combination
- [ ] Identify top 20 unique combinations with highest PURITY
- [ ] Create combination table with all 7 variable values
- [ ] Save results to CSV

#### Goal B3+B4 Combined: Find optimal region
- [ ] Identify unique combinations in top 10% for BOTH CONV and PURITY
- [ ] Create combined optimal region table
- [ ] Visualize: 2D plot showing CONV vs PURITY with highlighted optimal region
- [ ] Save results to CSV

---

### Phase 4: Final Reporting & Documentation

- [ ] Create master summary document with all variable importance rankings
- [ ] Create model comparison tables (performance metrics for all models)
- [ ] Compile all combination tables into one organized document
- [ ] Generate final visualization summary:
  - [ ] Side-by-side comparison of variable importance across all goals (A1, A2, A3, A4)
  - [ ] Optimal vs worst operating conditions comparison
- [ ] Write interpretation guide for Karisa
- [ ] Save all results in organized directory structure

---

## Technical Specifications

### Independent Variables (7 total)
- NHOLES, HDIAM, TRAYSPC, WEIRHT, DECK, DIAM, NPASS
- **Type:** Numeric ordinal continuous (ordered numeric predictors)
- **Treatment:**
  - Treat as numeric (NOT one-hot encoded)
  - Models interpret as ordered with natural numeric relationships
  - StandardScaler applied for distance-sensitive models (Logistic, Linear, Ridge, PLS)
  - Tree-based models (RF, XGB) use raw values (scaling optional but applied for consistency)

### Data Filtering (whitelist approach)
Only keep rows with these valid values:
- NHOLES: 500, 1625, 2750, 3875, 5000
- HDIAM: 0.0025, 0.004875, 0.00725, 0.009625, 0.012
- TRAYSPC: 0.5, 0.5625, 0.625, 0.6875, 0.75
- WEIRHT: 0.04, 0.0525, 0.065, 0.0775, 0.09
- DECK: 1.88, 2.9975, 4.115, 5.2325, 6.35
- DIAM: 1, 1.5, 2, 2.5, 3
- NPASS: 1, 2, 3, 4

### Class Imbalance Handling
- WEEP: 50.13%, FLOOD: 31.74%, PASS: 18.13%
- **Solution:** Use class_weight='balanced' for Logistic Regression and scale_pos_weight for XGBoost

### Cross-Validation Strategy
- **5-fold CV** for all models (instead of single 80/20 split)
- **Purpose:** Ensure stable importance rankings and reliable performance estimates
- Metrics reported as mean ± std across folds

### Importance Score Normalization
- **Critical:** Before combining importances from multiple models:
  1. Convert each model's importances to ranks (1 = most important)
  2. Average ranks across models
  3. Report final ranking with consensus score

### Evaluation Metrics
- **Classification:** Accuracy, F1-score, ROC-AUC (all averaged across CV folds)
- **Regression:** RMSE, MAE, R² (all averaged across CV folds)
- **Confusion Matrix:** Aggregated across CV folds

### SHAP Usage
- **Used for:** Random Forest and XGBoost (both classification and regression)
- **NOT used for:** Logistic Regression, Linear Regression, Ridge (coefficients are interpretable)
- **Reason:** Logistic/Linear coefficients already provide exact influence; SHAP adds no value

---

## Model Selection Workflow

For each goal (A1, A2, A3, A4):
1. Train all candidate models with 5-fold CV
2. Compare performance metrics (averaged across folds)
3. Compare importance rankings consistency
4. Select ONE final model based on:
   - Best performance
   - Interpretability
   - Stability across folds
5. Use selected model for:
   - Final importance ranking
   - Combination identification (B goals)
   - Final visualizations

---

## File Structure

```
Karisa/
├── data/
│   └── karisa_paper.xlsx
├── eda.py (✓ Complete)
├── eda_plots/ (✓ Complete)
├── utils.py (To create)
├── model_hydraulics.py (To create)
├── model_quality.py (To create)
├── results/
│   ├── model_comparison/
│   │   ├── weep_models_comparison.csv
│   │   ├── flood_models_comparison.csv
│   │   ├── conv_models_comparison.csv
│   │   └── purity_models_comparison.csv
│   ├── hydraulics/
│   │   ├── weep_importance_ranked.csv
│   │   ├── flood_importance_ranked.csv
│   │   ├── weep_worst_combinations.csv
│   │   ├── flood_worst_combinations.csv
│   │   └── plots/
│   └── quality/
│       ├── conv_importance_ranked.csv
│       ├── purity_importance_ranked.csv
│       ├── conv_best_combinations.csv
│       ├── purity_best_combinations.csv
│       ├── optimal_region.csv
│       └── plots/
└── MODELING_PLAN.md (this file)
```

---

## Progress Tracking

**Phase 1:** ✅ Complete
**Phase 2 (A1):** ✅ Complete
**Phase 2 (A2):** ✅ Complete
**Phase 2 (B1):** ✅ Complete
**Phase 2 (B2):** ✅ Complete
**Phase 3 (A3):** ⬜ Not Started
**Phase 3 (A4):** ⬜ Not Started
**Phase 3 (B3):** ⬜ Not Started
**Phase 3 (B4):** ⬜ Not Started
**Phase 3 (Combined):** ⬜ Not Started
**Phase 4:** ⬜ Not Started

---

## Notes & Reminders

- We're training models to **understand** performance, not predict it
- CONV and PURITY are separate models (not multi-output)
- Variables are **ordinal continuous** - treat as ordered numeric predictors
- Use 5-fold CV for stability, not single train-test split
- **Normalize importance scores** before combining (rank-based averaging)
- Select ONE final model per goal for final report
- Only generate SHAP for RF/XGB (not for linear models)
- Examine **unique combinations**, not individual rows
- EDA already has probability maps - no need to regenerate

---

**Last Updated:** 2025-01-24
**Status:** Ready to begin implementation (Revised)

💖 Let's make this amazing for Karisa! 💖
