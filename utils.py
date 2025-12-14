# -*- coding: utf-8 -*-
"""
Utility functions for Karisa's Chemical Engineering ML Project
Made with love for the smartest girl in the world
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import rankdata

# Independent variables - Karisa's carefully selected variables!
INDEPENDENT_VARS = ['NHOLES', 'HDIAM', 'TRAYSPC', 'WEIRHT', 'DECK', 'DIAM', 'NPASS']

# Valid values for each variable (whitelist - only keep these values)
VALID_VALUES = {
    'NHOLES': [500, 1625, 2750, 3875, 5000],
    'HDIAM': [0.0025, 0.004875, 0.00725, 0.009625, 0.012],
    'TRAYSPC': [0.5, 0.5625, 0.625, 0.6875, 0.75],
    'WEIRHT': [0.04, 0.0525, 0.065, 0.0775, 0.09],
    'DECK': [1.88, 2.9975, 4.115, 5.2325, 6.35],
    'DIAM': [1, 1.5, 2, 2.5, 3],
    'NPASS': [1, 2, 3, 4]
}


def load_data(data_path="data/AmAc_Tray.xlsx"):
    """
    Load both datasets from Excel file.

    Returns:
        df_full: Full dataset with all DESC values (PASS, WEEP, FLOOD)
        df_pass: Pass only dataset for quality metrics (CONV, PURITY)
    """
    data_path = Path(data_path)

    print("Loading data... (Karisa's amazing dataset!)")
    df_full = pd.read_excel(data_path, sheet_name="full_dataset")
    df_pass = pd.read_excel(data_path, sheet_name="pass_only")

    print(f"   Full dataset: {df_full.shape}")
    print(f"   Pass only dataset: {df_pass.shape}")
    print("   Data loaded successfully! You're brilliant, Karisa!")

    return df_full, df_pass


def filter_invalid_values(df_full, df_pass):
    """
    Keep only rows with valid values (whitelist filtering).

    Args:
        df_full: Full dataset
        df_pass: Pass only dataset

    Returns:
        df_full_filtered: Filtered full dataset
        df_pass_filtered: Filtered pass only dataset
    """
    print("\nFiltering to keep only valid values... (Karisa knows best!)")

    original_full = len(df_full)
    original_pass = len(df_pass)

    # Apply whitelist filters - keep only valid values
    for var, valid_vals in VALID_VALUES.items():
        if var in df_full.columns:
            df_full = df_full[df_full[var].isin(valid_vals)]
        if var in df_pass.columns:
            df_pass = df_pass[df_pass[var].isin(valid_vals)]

    # Fill missing DESC values with "FLOOD"
    if 'DESC' in df_full.columns:
        df_full['DESC'].fillna("FLOOD", inplace=True)

    print(f"   Full dataset: {original_full} -> {len(df_full)} rows (removed {original_full - len(df_full)})")
    print(f"   Pass dataset: {original_pass} -> {len(df_pass)} rows (removed {original_pass - len(df_pass)})")
    print("   Clean data for the smartest engineer!")

    return df_full, df_pass


def deduplicate_data(df_full, df_pass):
    """
    Remove duplicates based on independent variables only (NO column excluded).

    For duplicate combinations:
    - DESC: Take most common value (mode)
    - CONV/PURITY: Take mean

    Args:
        df_full: Full dataset
        df_pass: Pass only dataset

    Returns:
        df_full_dedup: Deduplicated full dataset
        df_pass_dedup: Deduplicated pass dataset
    """
    print("\nRemoving duplicates based on independent variables... (NO column excluded)")

    original_full = len(df_full)
    original_pass = len(df_pass)

    # Full dataset deduplication
    # Group by independent variables and aggregate
    agg_dict_full = {}
    for col in df_full.columns:
        if col in INDEPENDENT_VARS:
            agg_dict_full[col] = 'first'  # Keep the value (they're all the same within group)
        elif col == 'DESC':
            agg_dict_full[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Mode
        elif col != 'NO':  # Skip NO column
            agg_dict_full[col] = 'mean'  # Mean for numeric outcomes

    df_full_dedup = df_full.groupby(INDEPENDENT_VARS, as_index=False).agg(agg_dict_full)

    # Pass dataset deduplication
    agg_dict_pass = {}
    for col in df_pass.columns:
        if col in INDEPENDENT_VARS:
            agg_dict_pass[col] = 'first'
        elif col in ['CONV', 'PURITY']:
            agg_dict_pass[col] = 'mean'  # Average the outcomes
        elif col != 'NO':
            agg_dict_pass[col] = 'mean'

    df_pass_dedup = df_pass.groupby(INDEPENDENT_VARS, as_index=False).agg(agg_dict_pass)

    print(f"   Full dataset: {original_full} -> {len(df_full_dedup)} rows (removed {original_full - len(df_full_dedup)} duplicates)")
    print(f"   Pass dataset: {original_pass} -> {len(df_pass_dedup)} rows (removed {original_pass - len(df_pass_dedup)} duplicates)")
    print("   Deduplicated data ready!")

    return df_full_dedup, df_pass_dedup


def create_binary_targets(df_full):
    """
    Create binary targets for WEEP and FLOOD classification.

    Args:
        df_full: Full dataset with DESC column

    Returns:
        df_full: Dataset with added is_weep and is_flood columns
    """
    print("\nCreating binary targets... (I love you Karisa!)")

    if 'DESC' in df_full.columns:
        df_full['is_weep'] = (df_full['DESC'] == 'WEEP').astype(int)
        df_full['is_flood'] = (df_full['DESC'] == 'FLOOD').astype(int)

        print(f"   is_weep: {df_full['is_weep'].sum()} positive samples ({df_full['is_weep'].mean()*100:.2f}%)")
        print(f"   is_flood: {df_full['is_flood'].sum()} positive samples ({df_full['is_flood'].mean()*100:.2f}%)")
        print("   Binary targets ready! You've got this!")

    return df_full


def get_cv_splits(X, y, n_splits=5, stratified=True, random_state=42):
    """
    Get cross-validation splits.

    Args:
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds (default: 5)
        stratified: Whether to use stratified splits (for classification)
        random_state: Random seed

    Returns:
        cv: Cross-validation splitter
    """
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return cv


def get_scaler():
    """
    Get StandardScaler for feature scaling.

    Returns:
        scaler: StandardScaler instance
    """
    return StandardScaler()


def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan

    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model performance.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    return metrics


def aggregate_cv_metrics(metrics_list):
    """
    Aggregate metrics across CV folds.

    Args:
        metrics_list: List of metric dictionaries from each fold

    Returns:
        aggregated: Dictionary with mean ± std for each metric
    """
    aggregated = {}

    # Get all metric names (excluding confusion_matrix)
    metric_names = [k for k in metrics_list[0].keys() if k != 'confusion_matrix']

    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        aggregated[f'{metric_name}_mean'] = np.mean(values)
        aggregated[f'{metric_name}_std'] = np.std(values)

    # Aggregate confusion matrices if present
    if 'confusion_matrix' in metrics_list[0]:
        cms = np.array([m['confusion_matrix'] for m in metrics_list])
        aggregated['confusion_matrix'] = np.mean(cms, axis=0)

    return aggregated


def aggregate_cross_validate_results(cv_results):
    """
    Aggregate metrics from cross_validate output.

    Args:
        cv_results: Dictionary returned by cross_validate with 'test_scores' keys

    Returns:
        aggregated: Dictionary with mean ± std for each metric
    """
    aggregated = {}

    # Get all metric names (excluding time metrics and estimators)
    for key in cv_results.keys():
        if key.startswith('test_') or key.startswith('score'):
            # Remove 'test_' prefix if present
            metric_name = key.replace('test_', '').replace('score_', '')

            values = cv_results[key]
            if isinstance(values, (list, np.ndarray)):
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
            else:
                aggregated[f'{metric_name}'] = values

    return aggregated


def normalize_importance_to_ranks(importances_dict):
    """
    Convert importance scores from multiple models to ranks, then average.

    This ensures fair comparison across models with different scales.
    Rank 1 = most important variable.

    Args:
        importances_dict: Dictionary mapping model_name -> importance_array
                         Each importance_array has shape (n_features,)

    Returns:
        averaged_ranks: Array of averaged ranks (lower = more important)
    """
    print("\nNormalizing importance scores to ranks... (Science is beautiful, just like you!)")

    n_features = len(INDEPENDENT_VARS)
    all_ranks = []

    for model_name, importances in importances_dict.items():
        # Convert to ranks (1 = most important)
        # Use 'ordinal' method to handle ties consistently
        ranks = rankdata(-importances, method='ordinal')  # Negative because higher importance = lower rank
        all_ranks.append(ranks)
        print(f"   {model_name}: {ranks}")

    # Average ranks across models
    averaged_ranks = np.mean(all_ranks, axis=0)

    print(f"\n   Averaged ranks: {averaged_ranks}")

    return averaged_ranks


def get_unique_combinations(df, variables=None):
    """
    Get all unique combinations of variables from dataframe.

    Args:
        df: Dataframe
        variables: List of variable names (default: INDEPENDENT_VARS)

    Returns:
        unique_combos: Dataframe with unique combinations
    """
    if variables is None:
        variables = INDEPENDENT_VARS

    unique_combos = df[variables].drop_duplicates().reset_index(drop=True)

    print(f"\nFound {len(unique_combos)} unique combinations from {len(df)} rows (Your data is perfect, Karisa!)")

    return unique_combos


def create_output_directories():
    """
    Create directory structure for outputs.
    """
    print("\nCreating output directories... (Everything organized for the best engineer!)")

    directories = [
        "results",
        "results/model_comparison",
        "results/hydraulics",
        "results/hydraulics/plots",
        "results/quality",
        "results/quality/plots"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}/")

    print("   All directories ready! You're so organized, Karisa!")


def save_importance_ranking(importance_scores, ranks, output_path):
    """
    Save variable importance ranking to CSV.

    Args:
        importance_scores: Dictionary of {model_name: importance_array}
        ranks: Averaged ranks from normalize_importance_to_ranks
        output_path: Path to save CSV
    """
    # Create dataframe
    df_importance = pd.DataFrame({
        'Variable': INDEPENDENT_VARS,
        'Average_Rank': ranks
    })

    # Add individual model importances
    for model_name, scores in importance_scores.items():
        df_importance[f'{model_name}_Importance'] = scores

    # Sort by average rank
    df_importance = df_importance.sort_values('Average_Rank')

    # Save
    df_importance.to_csv(output_path, index=False)
    print(f"\nSaved importance ranking to: {output_path} (Great insights ahead!)")

    return df_importance


def save_model_comparison(results_dict, output_path):
    """
    Save model comparison results to CSV.

    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        output_path: Path to save CSV
    """
    rows = []

    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)

    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(output_path, index=False)

    print(f"\nSaved model comparison to: {output_path} (Amazing results for an amazing engineer!)")

    return df_comparison


# Helper message for Karisa
def print_phase_start(phase_name):
    """Print a beautiful phase start message."""
    print("\n" + "=" * 80)
    print(f"Starting: {phase_name}")
    print("=" * 80)


def print_phase_complete(phase_name):
    """Print a beautiful phase completion message."""
    print("\n" + "=" * 80)
    print(f"Completed: {phase_name}")
    print("=" * 80)
    print("You're doing amazing, Karisa!\n")


# ===================================================================
# GRAPH FORMATTING UTILITIES - For Paper Publication
# ===================================================================

# Variable name mapping for proper axis labels
VARIABLE_LABELS = {
    'NHOLES': 'Number of holes',
    'HDIAM': 'Hole Diameter (m)',
    'DIAM': 'Diameter (m)',
    'WEIRHT': 'Weir Height (m)',
    'TRAYSPC': 'Tray spacing (m)',
    'DECK': 'Deck Thickness (mm)',
    'NPASS': 'Number of Passes',
    'CONV': 'Conversion (%)',
    'PURITY': 'Purity (%)'
}

# Color scheme for DESC outcomes
DESC_COLORS = {
    'PASS': '#E6B824',    # Yellow
    'WEEP': '#0E1768',    # Dark Blue
    'FLOOD': '#E022C1'    # Magenta
}

def get_variable_label(var_name):
    """
    Get proper axis label for a variable name.

    Args:
        var_name: Variable name (e.g., 'NHOLES', 'CONV')

    Returns:
        Formatted label string (e.g., 'Number of holes', 'Conversion (%)')
    """
    return VARIABLE_LABELS.get(var_name, var_name)


def format_axis_for_paper(ax, xlabel=None, ylabel=None,
                          title=None, colorbar_label=None, cbar=None):
    """
    Format axes according to paper requirements:
    - Font: Arial
    - Axis labels: 16pt
    - Axis ticks: 14pt
    - No titles

    Args:
        ax: Matplotlib axis object
        xlabel: X-axis variable name (will be converted to proper label)
        ylabel: Y-axis variable name (will be converted to proper label)
        title: Ignored (titles removed for paper)
        colorbar_label: Label for colorbar if present
        cbar: Colorbar object
    """
    # Set font to Arial
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'

    # Remove title (for paper)
    ax.set_title('')

    # Set axis labels with proper formatting
    if xlabel is not None:
        label_text = get_variable_label(xlabel) if xlabel in VARIABLE_LABELS else xlabel
        ax.set_xlabel(label_text, fontsize=16, fontfamily='Arial')

    if ylabel is not None:
        label_text = get_variable_label(ylabel) if ylabel in VARIABLE_LABELS else ylabel
        ax.set_ylabel(label_text, fontsize=16, fontfamily='Arial')

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Format tick labels to use Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')

    # Format colorbar if present
    if cbar is not None and colorbar_label is not None:
        cbar.set_label(colorbar_label, fontsize=16, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=14)
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily('Arial')


def convert_to_percentage(df, columns=['CONV', 'PURITY']):
    """
    Convert decimal values to percentages (multiply by 100).

    Args:
        df: DataFrame
        columns: List of columns to convert (default: ['CONV', 'PURITY'])

    Returns:
        DataFrame with converted values
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col] * 100
    return df_copy


if __name__ == "__main__":
    # Test the utility functions
    print("=" * 80)
    print("Testing utility functions for Karisa's project")
    print("=" * 80)

    # Test data loading and filtering
    df_full, df_pass = load_data()
    df_full, df_pass = filter_invalid_values(df_full, df_pass)
    df_full = create_binary_targets(df_full)

    # Test directory creation
    create_output_directories()

    # Test unique combinations
    unique_combos = get_unique_combinations(df_full)

    print("\nAll utility functions tested successfully! (You're incredible, Karisa!)")
    print("Ready to start modeling for the smartest girl in the world!")
