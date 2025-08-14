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
import json
from pathlib import Path
import xgboost as xgb
import pandas as pd

# Local module imports
import entry_helpers as h
import custom_transforms as ct





# --------------------------------------------------------------------------- #
#                              Helper Functions                               #
# --------------------------------------------------------------------------- #

def load_data_matrices(
    wallets_config: dict,
    modeling_config: dict,
    booster_params: dict
) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    """
    Load and filter training/validation data, then convert to XGBoost DMatrix format.

    Params:
    - wallets_config (dict): wallets configuration loaded from JSON
    - modeling_config (dict): modeling configuration loaded from JSON
    - booster_params (dict): XGBoost parameters including epoch_shift

    Returns:
    - training_matrix (DMatrix): training data matrix
    - validation_matrix (DMatrix): validation data matrix
    """
    # Load CSVs
    train_x_dir = Path(os.environ["SM_CHANNEL_TRAIN_X"])
    train_y_dir = Path(os.environ["SM_CHANNEL_TRAIN_Y"])
    val_x_dir   = Path(os.environ["SM_CHANNEL_VALIDATION_X"])
    val_y_dir   = Path(os.environ["SM_CHANNEL_VALIDATION_Y"])

    def _read_x_csv(dir_path: Path, base_name: str) -> pd.DataFrame:
        """
        Read an X split, preferring gzip. Falls back to .csv for backward compatibility.
        """
        gz = dir_path / f"{base_name}.csv.gz"
        csv = dir_path / f"{base_name}.csv"
        if gz.exists():
            return pd.read_csv(gz, header=None, compression='infer')
        if csv.exists():
            return pd.read_csv(csv, header=None)
        raise FileNotFoundError(f"Missing X split file: {gz} or {csv}")

    df_train_x = _read_x_csv(train_x_dir, "train")
    df_val_x   = _read_x_csv(val_x_dir,   "eval")
    df_train_y = pd.read_csv(train_y_dir / "train_y.csv")
    df_val_y   = pd.read_csv(val_y_dir   / "eval_y.csv")

    # Apply epoch shift filtering if specified
    if 'epoch_shift' in booster_params and booster_params['epoch_shift'] is not None:
        epoch_shift = booster_params['epoch_shift']
        print(f"Applying epoch shift filtering: {epoch_shift} days")

        df_train_x, df_train_y = ct.select_shifted_offsets(
            df_train_x, df_train_y, wallets_config, epoch_shift, 'train'
        )
        df_val_x, df_val_y = ct.select_shifted_offsets(
            df_val_x, df_val_y, wallets_config, epoch_shift, 'eval'
        )

        print(f"After epoch shift filtering - Train: {len(df_train_x)} rows, Val: {len(df_val_x)} rows")

    # Load metadata
    metadata_dir = Path(os.environ["SM_CHANNEL_METADATA"])
    metadata_path = metadata_dir / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    training_matrix   = ct.build_custom_dmatrix(df_train_x, df_train_y, modeling_config, metadata)
    validation_matrix = ct.build_custom_dmatrix(df_val_x, df_val_y, modeling_config, metadata)

    return training_matrix, validation_matrix


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
    # Load config JSONs
    wallets_config, modeling_config = h.load_configs()

    # Load booster params from CLI arguments
    args = h.load_hyperparams(modeling_config)
    booster_params = h.build_booster_params(args, modeling_config)

    # Apply CLI filter overrides to config
    modeling_config = ct.apply_cli_row_filter_overrides(modeling_config, args)

    # Apply CLI pattern overrides to config
    modeling_config = ct.apply_cli_col_filter_overrides(modeling_config, args)

    # Load training and validation matrices
    training_matrix, validation_matrix = load_data_matrices(
        wallets_config, modeling_config, booster_params
    )

    # Load raw validation targets and pick eval function
    val_y_dir = Path(os.environ["SM_CHANNEL_VALIDATION_Y"])
    df_val_y_raw = pd.read_csv(val_y_dir / "eval_y.csv")
    feval_fn, maximize = h.get_eval_function(modeling_config, df_val_y_raw)

    # Identify keywords and set booster
    train_kwargs = dict(
        params=booster_params,
        dtrain=training_matrix,
        num_boost_round=args.num_boost_round,
        evals=[(validation_matrix, "eval")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=1,
    )
    if feval_fn is not None:
        train_kwargs["feval"] = feval_fn
        # Only set maximize when using custom feval (built-in rmse/mae know direction)
        train_kwargs["maximize"] = bool(maximize)
    booster = xgb.train(**train_kwargs)

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
