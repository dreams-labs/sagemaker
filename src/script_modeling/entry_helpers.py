import sys
import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb

import numpy as np
from sklearn.metrics import average_precision_score

# ---------------------------------
#     List of Valid Hyperparams
# ---------------------------------
def get_valid_hyperparameters(modeling_config) -> dict:
    """
    Return the complete set of valid hyperparameters, including dynamic filter params.

    Params:
    - modeling_config (dict, optional): If provided, includes dynamic filter params

    Returns:
    - dict: Complete hyperparameter type mapping
    """
    # Base hyperparameters that are always valid
    base_types = {
        'num_boost_round': int,
        'max_depth': int,
        'min_child_weight': int,
        'eta': float,
        'early_stopping_rounds': int,
        'colsample_bytree': float,
        'subsample': float,
        'scale_pos_weight': float,
        'alpha': float,
        'lambda': float,
        'gamma': float,
        'threshold': float,
    }

    # Add dynamic filter parameters if config provided
    if modeling_config:
        custom_filters = modeling_config.get('training', {}).get('custom_filters', {})
        for filter_config in custom_filters.values():
            cli_name = filter_config.get('cli')
            if cli_name:
                base_types[f'filter_{cli_name}_min'] = float
                base_types[f'filter_{cli_name}_max'] = float

    return base_types


# ---------------------------------
#     Model Pipeline Functions
# ---------------------------------
def load_hyperparams(modeling_config: dict) -> argparse.Namespace:
    """
    Parse hyperparameters injected by SageMaker for model training.

    Business logic:
    - Reads CLI arguments provided by the SageMaker training job.
    - Returns an argparse.Namespace with attributes:
      * eta (float): learning rate for XGBoost.
      * max_depth (int): tree depth.
      * subsample (float): row subsample ratio.
      * num_boost_round (int): number of boosting rounds.
      * early_stopping_rounds (int): rounds for early stopping based on eval set.
      * score_threshold (float): unused placeholder for downstream logic.

    Usage in entry scripts:
    ```python
    args = load_hyperparams()
    ```
    """
    parser = argparse.ArgumentParser()
    valid_hyperparams = get_valid_hyperparameters(modeling_config)

    # Detect which params were passed in CLI
    cli_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            name = arg.split('=')[0].lstrip('-')
            cli_args.add(name)

    # Add arguments only for hyperparameters present in CLI and known types
    for name, typ in valid_hyperparams.items():
        if name in cli_args:
            # Special-case bools for CLI parsing
            if typ is bool:
                parser.add_argument(f"--{name}", type=lambda x: x.lower() == 'true', required=True)
            else:
                parser.add_argument(f"--{name}", type=typ, required=True)

    return parser.parse_args()


def load_csv_as_dmatrix(csv_path: Path) -> xgb.DMatrix:
    """
    Load a CSV file into an XGBoost DMatrix.

    Business logic:
    - Assumes the first column is the binary label (target).
    - All remaining columns are numeric features.
    - Used both in single-model and temporal-CV entry points to prepare data.

    Usage in entry scripts:
    ```python
    dm = load_csv_as_dmatrix(Path("/opt/ml/input/data/train/train.csv"))
    ```
    """
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return xgb.DMatrix(features, label=labels)


def build_booster_params(args: argparse.Namespace) -> dict:
    """Construct XGBoost params by automatically unpacking all provided hyperparameters."""
    # Base parameters that are always included
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": 42,
    }

    # Convert args to dict and add all non-None hyperparameters
    args_dict = vars(args)

    for param_name, value in args_dict.items():
        if value is not None:
            params[param_name] = value

    return params


def print_metrics_and_save(booster: xgb.Booster, scores: list, model_dir: Path) -> None:
    """
    Print aggregated metrics and persist the best model.

    Business logic:
    - Computes and prints the mean PR-AUC across folds in a standardized format.
    - Saves the XGBoost model to the specified SM_MODEL_DIR for deployment.

    Usage in entry scripts:
    ```python
    print_metrics_and_save(best_booster, scores, Path(os.environ["SM_MODEL_DIR"]))
    ```
    """
    mean_pr = sum(scores) / len(scores) if scores else 0.0
    print(f"mean_cv_auc_pr={mean_pr:.6f}")
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_dir / "xgboost-model")




# ----------------------------------- #
#     Custom Evaluation Functions
# ----------------------------------- #
def eval_aucpr(preds, dtrain):
    """
    XGBoost feval: computes PR-AUC between labels and predictions.
    """
    labels = dtrain.get_label()
    return 'aucpr', average_precision_score(labels, preds)


def eval_top_quantile(df_val_y_raw: pd.DataFrame, metric_col: str, top_pct: float):
    """
    Returns an XGBoost feval that computes the mean value of `metric_col`
    over the top `top_pct` fraction of rows ranked by model score.
    """
    def eval_fn(preds, dtrain):  # pylint:disable=unused-argument
        # rank predictions descending
        idx_desc = np.argsort(preds)[::-1]
        # ensure at least one row
        n_top = max(int(len(preds) * top_pct), 1)
        # average the raw target values for the top-scoring rows
        raw_targets = df_val_y_raw[metric_col]
        mean_val = raw_targets.iloc[idx_desc[:n_top]].mean()
        return "top_quantile", float(mean_val)
    return eval_fn


# --------------------- Helper to select eval function by config --------------------- #

def get_eval_function(config: dict, df_val_y_raw: pd.DataFrame):
    """
    Select and return the appropriate XGBoost feval based on modeling config.
    """
    method = config['training']['eval_metric']
    custom_val = config['training'].get('custom_val', {})
    metric_col = custom_val.get('metric')
    if method == 'aucpr':
        return eval_aucpr
    elif method == 'top_quantile':
        top_pct = custom_val['top_quantile']
        return eval_top_quantile(df_val_y_raw, metric_col, top_pct)

    return eval_aucpr
