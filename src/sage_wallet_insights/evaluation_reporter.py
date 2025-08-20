import logging
import matplotlib.pyplot as plt
import matplotlib.cm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# pylint:disable=invalid-name  # X isn't lowercase
# pylint:disable=line-too-long # AI templates

# Set up logger at the module level
logger = logging.getLogger(__name__)


def summarize_multi_epoch_validation(evaluators_dict: dict) -> pd.DataFrame:
    """
    Aggregate validation metrics across all epoch shift evaluators.

    Params:
    - evaluators_dict (dict): {epoch_shift: evaluator} from build_all_epoch_shift_evaluators()

    Returns:
    - summary_df (DataFrame): Comprehensive metrics summary across all epochs
    """
    if not evaluators_dict:
        logger.warning("No evaluators provided for multi-epoch summary")
        return pd.DataFrame()

    # Determine model type from first evaluator
    first_evaluator = next(iter(evaluators_dict.values()))
    model_type = first_evaluator.modeling_config['model_type']

    summary_data = []

    for epoch_shift, evaluator in evaluators_dict.items():
        row_data = {
            'epoch_shift': epoch_shift,
            'model_type': model_type,
        }

        # Common metrics (both regression and classification)
        if hasattr(evaluator, 'validation_data_provided') and evaluator.validation_data_provided:
            val_metrics = evaluator.metrics.get('validation_metrics', {})

            # Core validation metrics
            row_data['val_samples'] = len(evaluator.y_validation) if evaluator.y_validation is not None else 0
            row_data['val_r2'] = val_metrics.get('r2', np.nan)
            row_data['val_rmse'] = val_metrics.get('rmse', np.nan)
            row_data['val_mae'] = val_metrics.get('mae', np.nan)
            row_data['val_spearman'] = val_metrics.get('spearman', np.nan)
            row_data['val_top1pct_mean'] = val_metrics.get('top1pct_mean', np.nan)

            # Return-based metrics (if available)
            row_data['val_ret_mean_overall'] = evaluator.metrics.get('val_ret_mean_overall', np.nan)
            row_data['val_wins_return_overall'] = evaluator.metrics.get('val_wins_return_overall', np.nan)
            row_data['val_ret_mean_top1'] = evaluator.metrics.get('val_ret_mean_top1', np.nan)
            row_data['val_ret_mean_top5'] = evaluator.metrics.get('val_ret_mean_top5', np.nan)
            row_data['val_wins_return_top1'] = evaluator.metrics.get('val_wins_return_top1', np.nan)
            row_data['val_wins_return_top5'] = evaluator.metrics.get('val_wins_return_top5', np.nan)

        # Classification-specific metrics
        if model_type == 'classification':
            row_data['val_roc_auc'] = evaluator.metrics.get('val_roc_auc', np.nan)
            row_data['val_accuracy'] = evaluator.metrics.get('val_accuracy', np.nan)
            row_data['val_precision'] = evaluator.metrics.get('val_precision', np.nan)
            row_data['val_recall'] = evaluator.metrics.get('val_recall', np.nan)
            row_data['val_f1'] = evaluator.metrics.get('val_f1', np.nan)

            # F-beta threshold metrics
            for beta in [0.1, 0.25, 0.5, 1.0, 2.0]:
                thr_key = f'f{beta}_thr'
                ret_key = f'val_ret_mean_f{beta}'
                wins_key = f'val_wins_ret_mean_f{beta}'

                row_data[f'f{beta}_threshold'] = evaluator.metrics.get(thr_key, np.nan)
                row_data[f'f{beta}_return_mean'] = evaluator.metrics.get(ret_key, np.nan)
                row_data[f'f{beta}_return_wins'] = evaluator.metrics.get(wins_key, np.nan)

            # Positive prediction metrics
            row_data['positive_predictions'] = evaluator.metrics.get('positive_predictions', np.nan)
            row_data['positive_pred_return'] = evaluator.metrics.get('positive_pred_return', np.nan)
            row_data['positive_pred_wins_return'] = evaluator.metrics.get('positive_pred_wins_return', np.nan)

        # Test set metrics (if available)
        if hasattr(evaluator, 'train_test_data_provided') and evaluator.train_test_data_provided:
            row_data['test_samples'] = evaluator.metrics.get('test_samples', 0)
            row_data['test_r2'] = evaluator.metrics.get('r2', np.nan)
            row_data['test_rmse'] = evaluator.metrics.get('rmse', np.nan)
            row_data['test_mae'] = evaluator.metrics.get('mae', np.nan)

            if model_type == 'classification':
                row_data['test_roc_auc'] = evaluator.metrics.get('roc_auc', np.nan)
                row_data['test_accuracy'] = evaluator.metrics.get('accuracy', np.nan)
                row_data['test_precision'] = evaluator.metrics.get('precision', np.nan)
                row_data['test_recall'] = evaluator.metrics.get('recall', np.nan)
                row_data['test_f1'] = evaluator.metrics.get('f1', np.nan)

        summary_data.append(row_data)

    # Create DataFrame and compute summary statistics
    summary_df = pd.DataFrame(summary_data)

    # Sort by epoch_shift for consistent ordering
    summary_df = summary_df.sort_values('epoch_shift').reset_index(drop=True)

    return summary_df


def print_validation_summary_report(summary_df: pd.DataFrame):
    """
    Print formatted validation summary report.

    Params:
    - summary_df (DataFrame): Output from summarize_multi_epoch_validation()
    """
    if summary_df.empty:
        print("No validation summary data available")
        return

    model_type = summary_df['model_type'].iloc[0]
    n_epochs = len(summary_df)

    print("Multi-Epoch Validation Summary")
    print("=" * 50)
    print(f"Model Type: {model_type.title()}")
    print(f"Epochs Analyzed: {n_epochs}")
    print(f"Epoch Shifts: {summary_df['epoch_shift'].tolist()}")
    print()

    # Core validation metrics summary
    validation_cols = ['val_r2', 'val_rmse', 'val_spearman', 'val_top1pct_mean']
    if model_type == 'classification':
        validation_cols.extend(['val_roc_auc', 'val_accuracy', 'val_f1'])

    # Filter to columns that exist and have non-NaN data
    available_cols = [col for col in validation_cols if col in summary_df.columns and not summary_df[col].isna().all()]

    if available_cols:
        print("Validation Performance Distribution")
        print("-" * 40)

        for col in available_cols:
            values = summary_df[col].dropna()
            if len(values) > 0:
                metric_name = col.replace('val_', '').replace('_', ' ').title()
                print(f"{metric_name:<20} | Mean: {values.mean():.3f} | Std: {values.std():.3f} | Range: {values.min():.3f}-{values.max():.3f}")
        print()

    # Return-based performance (most important for economic validation)
    return_cols = ['val_ret_mean_overall', 'val_ret_mean_top1', 'val_ret_mean_top5']
    available_return_cols = [col for col in return_cols if col in summary_df.columns and not summary_df[col].isna().all()]

    if available_return_cols:
        print("Return Performance Distribution")
        print("-" * 40)

        for col in available_return_cols:
            values = summary_df[col].dropna()
            if len(values) > 0:
                metric_name = col.replace('val_ret_mean_', '').replace('_', ' ').title()
                print(f"{metric_name:<20} | Mean: {values.mean():.3f} | Std: {values.std():.3f} | Range: {values.min():.3f}-{values.max():.3f}")
        print()

    # Best and worst performing epochs
    if 'val_r2' in summary_df.columns and not summary_df['val_r2'].isna().all():
        best_epoch = summary_df.loc[summary_df['val_r2'].idxmax(), 'epoch_shift']
        worst_epoch = summary_df.loc[summary_df['val_r2'].idxmin(), 'epoch_shift']
        print(f"Best R² Performance: Epoch {best_epoch} ({summary_df['val_r2'].max():.3f})")
        print(f"Worst R² Performance: Epoch {worst_epoch} ({summary_df['val_r2'].min():.3f})")

    if 'val_ret_mean_top1' in summary_df.columns and not summary_df['val_ret_mean_top1'].isna().all():
        best_return_epoch = summary_df.loc[summary_df['val_ret_mean_top1'].idxmax(), 'epoch_shift']
        worst_return_epoch = summary_df.loc[summary_df['val_ret_mean_top1'].idxmin(), 'epoch_shift']
        print(f"Best Top 1% Returns: Epoch {best_return_epoch} ({summary_df['val_ret_mean_top1'].max():.3f})")
        print(f"Worst Top 1% Returns: Epoch {worst_return_epoch} ({summary_df['val_ret_mean_top1'].min():.3f})")


def analyze_epoch_consistency(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze consistency of key metrics across epoch shifts.

    Params:
    - summary_df (DataFrame): Output from summarize_multi_epoch_validation()

    Returns:
    - consistency_df (DataFrame): Coefficient of variation for key metrics
    """
    if summary_df.empty:
        return pd.DataFrame()

    # Key metrics to analyze for consistency
    key_metrics = ['val_r2', 'val_rmse', 'val_spearman', 'val_top1pct_mean', 'val_ret_mean_top1']

    model_type = summary_df['model_type'].iloc[0]
    if model_type == 'classification':
        key_metrics.extend(['val_roc_auc', 'val_f1'])

    consistency_data = []

    for metric in key_metrics:
        if metric in summary_df.columns:
            values = summary_df[metric].dropna()
            if len(values) > 1:  # Need at least 2 values for std calculation
                mean_val = values.mean()
                std_val = values.std()
                cv = std_val / abs(mean_val) if mean_val != 0 else np.inf

                consistency_data.append({
                    'metric': metric.replace('val_', '').replace('_', ' ').title(),
                    'mean': mean_val,
                    'std': std_val,
                    'coefficient_variation': cv,
                    'n_epochs': len(values),
                    'consistency_score': 1 / (1 + cv)  # Higher is more consistent
                })

    consistency_df = pd.DataFrame(consistency_data)
    if not consistency_df.empty:
        consistency_df = consistency_df.sort_values('consistency_score', ascending=False)

    return consistency_df


def plot_multi_epoch_validation_curves(evaluators_dict: dict, display: bool = True):
    """
    Plot ROC and PR curves for all epoch shifts on validation data,
    plus aggregated return curves by prediction score.

    Params:
    - evaluators_dict (dict): {epoch_shift: evaluator} from build_all_epoch_shift_evaluators()
    - display (bool): If True, show plots directly; if False, return the figure

    Returns:
    - fig: matplotlib figure (if display=False)
    """
    if not evaluators_dict:
        logger.warning("No evaluators provided for multi-epoch curve plotting")
        return None

    # Check if we have classification models with validation data
    first_evaluator = next(iter(evaluators_dict.values()))
    model_type = first_evaluator.modeling_config['model_type']

    if model_type != 'classification':
        logger.warning("ROC/PR curves are only available for classification models")
        return None

    # Filter to evaluators with validation data
    valid_evaluators = {
        epoch_shift: evaluator for epoch_shift, evaluator in evaluators_dict.items()
        if (hasattr(evaluator, 'validation_data_provided') and evaluator.validation_data_provided and
            hasattr(evaluator, 'y_validation_pred_proba') and evaluator.y_validation_pred_proba is not None)
    }

    if not valid_evaluators:
        logger.warning("No evaluators with validation probability predictions found")
        return None

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Generate colors for each epoch shift
    n_epochs = len(valid_evaluators)
    colors = matplotlib.cm.viridis(np.linspace(0, 1, n_epochs))  #pylint:disable=no-member

    # Sort evaluators by epoch_shift for consistent ordering
    sorted_evaluators = dict(sorted(valid_evaluators.items()))

    # Storage for legend entries and overall averages
    roc_legend_entries = []
    pr_legend_entries = []
    all_roc_curves = []
    all_pr_data = []  # Store (y_val, y_val_proba) for threshold-based averaging
    all_return_curves = []

    for (epoch_shift, evaluator), color in zip(sorted_evaluators.items(), colors):
        try:
            # Extract validation data
            y_val = evaluator.y_validation
            y_val_proba = evaluator.y_validation_pred_proba

            # Store data for threshold-based PR averaging
            all_pr_data.append((y_val, y_val_proba))

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_val, y_val_proba)
            roc_auc = auc(fpr, tpr)

            # Store ROC curve for averaging
            all_roc_curves.append((fpr, tpr))

            # Plot ROC curve
            ax1.plot(fpr, tpr, color=color, linewidth=2, alpha=0.8,
                    label=f'Epoch {epoch_shift} (AUC: {roc_auc:.3f})')

            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
            baseline = y_val.mean()  # Positive class prevalence

            # Normalize precision by subtracting baseline
            precision_normalized = precision - baseline
            pr_auc = auc(recall, precision)
            pr_auc_normalized = pr_auc - baseline

            # Plot normalized PR curve (skip first point to avoid misleading spike)
            ax2.plot(recall[1:], precision_normalized[1:], color=color, linewidth=2, alpha=0.8,
                    label=f'Epoch {epoch_shift} (Δ-AUC: {pr_auc_normalized:.3f})')

            # Extract and plot validation return curve
            bucket_df = evaluator.compute_score_buckets(n_buckets=20)
            if not bucket_df.empty:
                # Store return curve for averaging
                all_return_curves.append((bucket_df["score_mid"].values, bucket_df["wins_return"].values))

                ax3.plot(bucket_df["score_mid"], bucket_df["wins_return"],
                        color=color, linewidth=2, alpha=0.8,
                        label=f'Epoch {epoch_shift}')

            roc_legend_entries.append((epoch_shift, roc_auc))
            pr_legend_entries.append((epoch_shift, pr_auc_normalized))

        except Exception as e:
            logger.warning(f"Could not plot curves for epoch_shift {epoch_shift}: {e}")
            continue

    # Calculate and plot overall averages
    if all_roc_curves:
        # Average ROC curve using interpolation
        common_fpr = np.linspace(0, 1, 100)
        avg_tpr = np.zeros_like(common_fpr)

        for fpr, tpr in all_roc_curves:
            avg_tpr += np.interp(common_fpr, fpr, tpr)
        avg_tpr /= len(all_roc_curves)

        avg_roc_auc = auc(common_fpr, avg_tpr)
        ax1.plot(common_fpr, avg_tpr, color='white', linewidth=3, linestyle=':', alpha=0.9,
                label=f'Overall Average (AUC: {avg_roc_auc:.3f})')

    # For PR curves, use threshold-based averaging
    if all_pr_data:
        # Combine all validation data
        all_y_val = np.concatenate([y_val for y_val, _ in all_pr_data])
        all_y_proba = np.concatenate([y_proba for _, y_proba in all_pr_data])

        # Calculate overall baseline
        overall_baseline = all_y_val.mean()

        # Calculate PR curve on combined data
        combined_precision, combined_recall, _ = precision_recall_curve(all_y_val, all_y_proba)
        combined_precision_norm = combined_precision - overall_baseline
        combined_pr_auc = auc(combined_recall, combined_precision_norm)

        # Plot combined average curve
        ax2.plot(combined_recall[1:], combined_precision_norm[1:], color='white', linewidth=3, linestyle=':', alpha=0.9,
                label=f'Overall Average (Δ-AUC: {combined_pr_auc:.3f})')

    if all_return_curves:
        # Average return curves using common score grid
        all_scores = np.concatenate([scores for scores, _ in all_return_curves])
        common_scores = np.linspace(all_scores.min(), all_scores.max(), 50)
        avg_returns = np.zeros_like(common_scores)
        valid_curves = 0

        for scores, returns in all_return_curves:
            if len(scores) > 1 and len(returns) > 1:
                avg_returns += np.interp(common_scores, scores, returns)
                valid_curves += 1

        if valid_curves > 0:
            avg_returns /= valid_curves
            ax3.plot(common_scores, avg_returns, color='white', linewidth=3, linestyle=':', alpha=0.9,
                    label='Overall Average')

    # ROC plot formatting
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - All Epoch Shifts (Validation)')
    ax1.grid(True, linestyle=":", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=10)

    # PR plot formatting
    # Add zero reference line (baseline performance)
    ax2.axhline(0, linestyle='--', linewidth=1, alpha=0.6,
               color='gray', label='Baseline (Δ=0)')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision - Baseline')
    ax2.set_title('Normalized Precision-Recall Curves - All Epoch Shifts (Validation)')
    ax2.grid(True, linestyle=":", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    # Return curves plot formatting
    # Add overall mean return reference line if we have data
    if valid_evaluators:
        # Get overall mean from first evaluator as reference
        first_eval = next(iter(valid_evaluators.values()))
        if hasattr(first_eval, 'validation_target_vars_df') and first_eval.validation_target_vars_df is not None:
            target_var = first_eval.modeling_config["target_variable"]
            overall_mean = first_eval.validation_target_vars_df[target_var].mean()
            ax3.axhline(overall_mean, linestyle='--', linewidth=1, alpha=0.6,
                       color='gray', label='Overall Mean Return')

    ax3.set_xlabel('Prediction Score')
    ax3.set_ylabel('Winsorized Return')
    ax3.set_title('Validation Return Curves - All Epoch Shifts')
    ax3.grid(True, linestyle=":", alpha=0.3)
    ax3.legend(loc="upper left", fontsize=10)


def plot_validation_returns_distribution(evaluators_dict: dict, display: bool = True):
    """
    Plot distribution of validation returns across all epoch shifts.

    Params:
    - evaluators_dict (dict): {epoch_shift: evaluator} from build_all_epoch_shift_evaluators()
    - display (bool): If True, show plots directly; if False, return the figure

    Returns:
    - fig: matplotlib figure (if display=False)
    """
    if not evaluators_dict:
        logger.warning("No evaluators provided for returns distribution plotting")
        return None

    # Filter to evaluators with validation return data
    valid_evaluators = {
        epoch_shift: evaluator for epoch_shift, evaluator in evaluators_dict.items()
        if (hasattr(evaluator, 'validation_data_provided') and evaluator.validation_data_provided and
            hasattr(evaluator, 'validation_target_vars_df') and evaluator.validation_target_vars_df is not None)
    }

    if not valid_evaluators:
        logger.warning("No evaluators with validation return data found")
        return None

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Collect data for all epochs
    epoch_returns_data = []
    top1_returns = []
    top5_returns = []
    overall_returns = []

    first_evaluator = next(iter(valid_evaluators.values()))
    target_var = first_evaluator.modeling_config['target_variable']

    for epoch_shift, evaluator in sorted(valid_evaluators.items()):
        # Get return data
        val_ret_mean_overall = evaluator.metrics.get('val_ret_mean_overall', np.nan)
        val_ret_mean_top1 = evaluator.metrics.get('val_ret_mean_top1', np.nan)
        val_ret_mean_top5 = evaluator.metrics.get('val_ret_mean_top5', np.nan)

        if not np.isnan(val_ret_mean_overall):
            overall_returns.append((epoch_shift, val_ret_mean_overall))
        if not np.isnan(val_ret_mean_top1):
            top1_returns.append((epoch_shift, val_ret_mean_top1))
        if not np.isnan(val_ret_mean_top5):
            top5_returns.append((epoch_shift, val_ret_mean_top5))

        # Collect individual return distributions if available
        if hasattr(evaluator, 'validation_target_vars_df'):
            returns = evaluator.validation_target_vars_df[target_var].dropna()
            epoch_returns_data.append((epoch_shift, returns))

    # Plot 1: Return performance by epoch shift
    if overall_returns:
        epochs, overall_vals = zip(*overall_returns)
        ax1.plot(epochs, overall_vals, 'o-', linewidth=2, markersize=6,
                label='Overall Average', color='blue')

    if top5_returns:
        epochs, top5_vals = zip(*top5_returns)
        ax1.plot(epochs, top5_vals, 's-', linewidth=2, markersize=6,
                label='Top 5% Scores', color='orange')

    if top1_returns:
        epochs, top1_vals = zip(*top1_returns)
        ax1.plot(epochs, top1_vals, '^-', linewidth=2, markersize=6,
                label='Top 1% Scores', color='red')

    ax1.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax1.set_xlabel('Epoch Shift')
    ax1.set_ylabel(f'Mean {target_var}')
    ax1.set_title('Validation Return Performance by Epoch')
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.3)

    # Plot 2: Return consistency (coefficient of variation)
    metrics_data = {
        'Overall': [val for _, val in overall_returns],
        'Top 5%': [val for _, val in top5_returns],
        'Top 1%': [val for _, val in top1_returns]
    }

    cv_data = []
    for metric_name, values in metrics_data.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
            cv_data.append((metric_name, cv))

    if cv_data:
        names, cvs = zip(*cv_data)
        bars = ax2.bar(names, cvs, color=['blue', 'orange', 'red'], alpha=0.7)
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Return Consistency Across Epochs\n(Lower = More Consistent)')
        ax2.grid(True, linestyle=":", alpha=0.3)

        # Add value labels on bars
        for bar, cv in zip(bars, cvs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cv:.3f}', ha='center', va='bottom')

    # Plot 3: Distribution of returns across all epochs (if available)
    if epoch_returns_data and len(epoch_returns_data) > 1:
        all_returns = []
        for epoch_shift, returns in epoch_returns_data[:5]:  # Limit to first 5 epochs
            all_returns.extend(returns.values)
            sns.kdeplot(data=returns, ax=ax3, label=f'Epoch {epoch_shift}', alpha=0.7)

        ax3.axvline(np.mean(all_returns), color='black', linestyle='--',
                   label=f'Overall Mean ({np.mean(all_returns):.3f})')
        ax3.set_xlabel(f'{target_var}')
        ax3.set_ylabel('Density')
        ax3.set_title('Return Distributions by Epoch (Sample)')
        ax3.legend()
        ax3.grid(True, linestyle=":", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Individual return distributions\nnot available',
                ha='center', va='center', transform=ax3.transAxes)

    # Plot 4: Summary statistics table
    ax4.axis('off')

    if overall_returns and top1_returns:
        summary_text = []
        summary_text.append("Validation Returns Summary")
        summary_text.append("=" * 30)
        summary_text.append(f"Epochs Analyzed: {len(overall_returns)}")
        summary_text.append("")

        # Overall returns stats
        overall_vals = [val for _, val in overall_returns]
        summary_text.append("Overall Average Returns:")
        summary_text.append(f"  Mean: {np.mean(overall_vals):.3f}")
        summary_text.append(f"  Std:  {np.std(overall_vals):.3f}")
        summary_text.append(f"  Range: {np.min(overall_vals):.3f} to {np.max(overall_vals):.3f}")
        summary_text.append("")

        # Top 1% returns stats
        top1_vals = [val for _, val in top1_returns]
        summary_text.append("Top 1% Score Returns:")
        summary_text.append(f"  Mean: {np.mean(top1_vals):.3f}")
        summary_text.append(f"  Std:  {np.std(top1_vals):.3f}")
        summary_text.append(f"  Range: {np.min(top1_vals):.3f} to {np.max(top1_vals):.3f}")
        summary_text.append("")

        # Best/worst epochs
        best_overall_idx = np.argmax(overall_vals)
        worst_overall_idx = np.argmin(overall_vals)
        best_top1_idx = np.argmax(top1_vals)

        summary_text.append("Best Performance:")
        summary_text.append(f"  Overall: Epoch {overall_returns[best_overall_idx][0]} ({overall_vals[best_overall_idx]:.3f})")
        summary_text.append(f"  Top 1%:  Epoch {top1_returns[best_top1_idx][0]} ({top1_vals[best_top1_idx]:.3f})")
        summary_text.append("")
        summary_text.append("Worst Overall:")
        summary_text.append(f"  Epoch {overall_returns[worst_overall_idx][0]} ({overall_vals[worst_overall_idx]:.3f})")

        ax4.text(0.05, 0.95, '\n'.join(summary_text), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if display:
        plt.show()
        return None
    return fig
