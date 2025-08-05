import sys
import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb


HYPERPARAMETER_TYPES = {
    'num_boost_round': int,
    'max_depth': int,
    'min_child_weight': int,
    'eta': float,
    'early_stopping_rounds': int,
    'colsample_bytree': float,
    'subsample': float,
    'scale_pos_weight': float,
    'alpha': float,           # L1 regularization
    'lambda': float           # L2 regularization
}

def load_hyperparams() -> argparse.Namespace:
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

    # Detect which params were passed in CLI
    cli_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            name = arg.split('=')[0].lstrip('-')
            cli_args.add(name)

    # Add arguments only for hyperparameters present in CLI and known types
    for name, typ in HYPERPARAMETER_TYPES.items():
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
