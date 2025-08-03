import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb


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
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Learning rate for XGBoost (eta)")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Maximum tree depth for XGBoost")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Subsample ratio of the training instance")
    parser.add_argument("--num_boost_round", type=int, default=500,
                        help="Number of boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, default=30,
                        help="Rounds for early stopping on validation set")
    parser.add_argument("--colsample_bytree", type=float, default=0.9,
                        help="Rounds for early stopping on validation set")
    parser.add_argument("--scale_pos_weight", type=float, default=0.9,
                        help="Rounds for early stopping on validation set")
    parser.add_argument("--score_threshold", type=float, default=0.80,
                        help="Placeholder threshold for business logic filtering")
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
    """
    Construct the parameter dict for XGBoost training from parsed args.

    Business logic:
    - Sets objective to binary logistic for wallet classification.
    - Uses "aucpr" as the evaluation metric to focus on top-ranked predictions.
    - Maps CLI args to XGBoost params.

    Usage in entry scripts:
    ```python
    params = build_booster_params(args)
    ```
    """
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "seed": 42,
    }


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
