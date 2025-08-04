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
import traceback
import os
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import average_precision_score

# Local module imports
import entry_helpers as h



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

    # Debug: only print dir listings if expected CSVs are missing
    train_dir = Path("/opt/ml/input/data/train")
    val_dir = Path("/opt/ml/input/data/validation")
    train_csv = train_dir / "train.csv"
    val_csv = val_dir / "eval.csv"
    if not (train_csv.exists() and val_csv.exists()):
        print("Train directory contents:", os.listdir(train_dir))
        print("Validation directory contents:", os.listdir(val_dir))

    # Load CSVs in XGBoost's dmatrix format
    training_matrix = h.load_csv_as_dmatrix(train_csv)
    validation_matrix = h.load_csv_as_dmatrix(val_csv)

    # Define hyperparameters from the parse arguments
    args = h.load_hyperparams()
    booster_params = h.build_booster_params(args)

    # Define custom PR-AUC evaluation metric
    def eval_aucpr(preds, dtrain):
        labels = dtrain.get_label()
        return 'eval_aucpr', average_precision_score(labels, preds)

    # Train with early stopping and per-round PR-AUC logging
    booster = xgb.train(
        params=booster_params,
        dtrain=training_matrix,
        num_boost_round=args.num_boost_round,
        evals=[(validation_matrix, "eval")],
        early_stopping_rounds=args.early_stopping_rounds,
        feval=eval_aucpr,
        maximize=True,
        verbose_eval=1,
    )

    # Compute PR-AUC on validation
    y_val_predictions = booster.predict(validation_matrix)
    y_val_actuals = validation_matrix.get_label()
    pr_auc = average_precision_score(y_val_actuals, y_val_predictions)

    # Emit metric for SageMaker’s regex parser.
    print(f"validation:aucpr={pr_auc:.6f}")

    # Persist model artifacts to relative paths within the container
    model_output_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_output_dir / "xgboost-model")


# Run the script when executed inside the container
if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise