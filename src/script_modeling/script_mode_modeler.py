"""
Custom script-mode trainer for SageMaker XGBoost.

Business logic
--------------
1. Read the **train** and **validation** CSVs that SageMaker mounts at
   /opt/ml/input/data/{train|validation}/.
2. Fit a binary-logistic XGBoost model using the hyper-parameters SageMaker
   injects via CLI arguments.
3. Evaluate **PR-AUC** on the validation set and print it in
   `label=value` form so SageMaker Hyperparameter Tuner can optimise it.
4. Persist the trained booster to SM_MODEL_DIR so the default inference
   container can serve predictions.

The script is intentionally minimal — once validated, we will extend the
“Evaluate” step to loop over multiple date-suffix folds and/or compute
mean-return-above-threshold.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score



# --------------------------------------------------------------------------- #
#                               Helper Functions                              #
# --------------------------------------------------------------------------- #
def load_hyperparams_from_cli() -> argparse.Namespace:
    """
    Parse command-line arguments injected by SageMaker. This allows us to read
     the CLI and extract specified lines into a lightweight container (Namespace).
     If any of our arguments don't already exist, the Parser will add them using
     the default value.

    Returns
    - argparse.Namespace: a lightweight container that stores attributes
    """
    # This Parser can read the command line code within the SageMaker container
    parser = argparse.ArgumentParser()

    # The Parser will look for these lines in the CLI, and will enter them with the
    #  default values if they don't already exist.

    # ----- core XGBoost hyper-parameters ------------ #
    parser.add_argument("--eta", type=float, default=0.1)  # eta = learning rate
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=30)

    # ----- business-logic hyper-parameters ---------- #
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.80,
        help="Wallet-selection cut-off; unused in this MVP but "
             "kept for interface consistency.",
    )

    # Command to read the CLI, match the add_argument flags, and returns an
    #  argparse.Namespace, which is a lightweight container that stores attributes
    return parser.parse_args()


def load_csv_as_dmatrix(csv_path: Path) -> xgb.DMatrix:
    """
    Convert SageMaker channel CSV into `xgboost.DMatrix`.

    Parameters
    ----------
    csv_path : Path
        File containing data with **target** in column 0 and features in
        remaining columns.

    Returns
    -------
    xgb.DMatrix
        Matrix ready for XGBoost training.
    """
    dataframe = pd.read_csv(csv_path, header=None)
    target_vector = dataframe.iloc[:, 0].values
    feature_matrix = dataframe.iloc[:, 1:].values

    # Label is XGBoost's term for target variable
    return xgb.DMatrix(feature_matrix, label=target_vector)



# --------------------------------------------------------------------------- #
#                                Main Routine                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Train one fold of the wallet-classifier in SageMaker *script-mode*.

    The routine reads the pre-mounted train/validation CSVs, fits an XGBoost
    binary-logistic model with early stopping, prints the validation PR-AUC
    in `label=value` form for the HPO tuner, and saves the trained booster
    to `$SM_MODEL_DIR` so the default inference container can serve it. All
    filepaths are relative to the container only and have no impact on S3 or
    local directories.
    """

    # Loads training and validation data from their relative local path that the
    #  container has copied them to

    print("Train files:", os.listdir("/opt/ml/input/data/train/"))
    print("Validation files:", os.listdir("/opt/ml/input/data/validation/"))


    train_csv = Path("/opt/ml/input/data/train/train.csv")
    val_csv = Path("/opt/ml/input/data/validation/eval.csv")
    training_matrix = load_csv_as_dmatrix(train_csv)
    validation_matrix = load_csv_as_dmatrix(val_csv)

    # Define hyperparameters from the parse arguments
    args = load_hyperparams_from_cli()
    booster_params: dict[str, float | int | str] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",     # built-in PR-AUC for training progress
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "seed": 42,
    }

    # Train with early stopping
    booster = xgb.train(
        params=booster_params,
        dtrain=training_matrix,
        num_boost_round=args.num_boost_round,
        evals=[(validation_matrix, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    # Compute PR-AUC on validation
    y_val_predictions = booster.predict(validation_matrix)
    y_val_actuals = validation_matrix.get_label()
    pr_auc = average_precision_score(y_val_actuals, y_val_predictions)

    # Emit metric for SageMaker’s regex parser.
    print(f"validation:cv_auc_pr={pr_auc:.6f}")

    # Persist model artifacts to relative paths within the container
    model_output_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_output_dir / "xgboost-model")


# Run the script when executed inside the container
if __name__ == "__main__":
    main()
