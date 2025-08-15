import sys
import logging
from typing import Tuple, Union
from pathlib import Path
import json
import pandas as pd
import numpy as np

# pylint:disable=wrong-import-position
# ensure script_modeling directory is on path for its module
script_modeling_dir = Path(__file__).resolve().parents[1] / "script_modeling"
sys.path.insert(0, str(script_modeling_dir))
from custom_transforms import apply_row_filters, preprocess_custom_labels, select_shifted_offsets, identify_offset_ints
import sage_wallet_insights.evaluation_orchestrator as eo

# Import from data-science repo
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
import wallet_insights.model_evaluation as wime
import utils as u
from utils import ConfigError

# pylint:disable=invalid-name  # X isn't lowercase

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Local control-flow exception used to skip epochs when filters remove all rows
class SkipEpochEvaluation(Exception):
    """Signal that this epoch has no rows after filtering and should be skipped."""
    pass


# --------------------------
#      Primary Interface
# --------------------------
@u.timing_decorator
def create_concatenated_sagemaker_evaluator(
    wallets_config: dict,
    modeling_config: dict,
    model_uri: str,
    y_test_pred: pd.Series,
    y_test: pd.DataFrame,
    y_val_pred: pd.Series = None,
    y_val: pd.DataFrame = None,
    epoch_shift: int = 0,
    y_train: pd.Series = None,
    X_train: pd.DataFrame = None
) -> Union[wime.RegressorEvaluator, wime.ClassifierEvaluator]:
    """
    Create a SageMaker evaluator for concatenated model results with optional validation set.

    Params:
    - wallets_config (dict): Configuration for training data paths
    - modeling_config (dict): Configuration for model parameters
    - model_uri (str): Used as model_id
    - y_test_pred (Series): Predicted values for the concatenated test set
    - y_test (DataFrame): Single-column actual target for the concatenated test set
    - y_val_pred (Series, optional): Predicted values for validation set
    - y_val (DataFrame, optional): Single-column actual target for the validation set
    - epoch_shift (int, optional): How many days to shift the base offsets in wallets_config
    - y_train (Series, optional): Pre-loaded training target data
    - X_train (DataFrame, optional): Pre-loaded training feature data

    Returns:
    - RegressorEvaluator or ClassifierEvaluator: Configured evaluator
      (May return None if all rows are filtered out for the given epoch.)
    """
    # Apply custom transforms and get filtered data
    try:
        (y_test_final, X_test_final, y_test_pred_final, y_val_final, X_val_final, y_val_pred_final,
         row_mask_val, epoch_mask_val) = \
            _apply_custom_transforms_to_concatenated_data(
                wallets_config, modeling_config, y_test_pred, y_test, y_val_pred, y_val, epoch_shift
            )
    except SkipEpochEvaluation:
        logger.warning("Skipping evaluation for epoch_shift=%s (all rows filtered out).", epoch_shift)
        return None

    # Load train data - use pre-loaded if provided, otherwise load fresh
    if y_train is not None and X_train is not None:
        train_y = y_train
        train_X = X_train
    else:
        train_y, train_X = eo.load_concatenated_features(wallets_config)

    y_test_series = y_test_final
    y_test_pred = y_test_pred_final

    # Set validation data
    validation_provided = y_val_final is not None
    y_val_series = y_val_final if validation_provided else None
    y_val_pred = y_val_pred_final if validation_provided else None

    # Continuous target variable (used for returns plots)
    if 'target' not in modeling_config or 'target_var' not in modeling_config['target']:
        raise ConfigError("Missing 'target.target_var' in modeling_config; required for returns plotting.")
    cont_target_var = modeling_config['target']['target_var']
    wime_modeling_config = {
        "target_variable": cont_target_var,
        "target_var_min_threshold": modeling_config['target']['classification']['threshold'],
        "model_type": modeling_config["training"]["model_type"],
        "returns_winsorization": 0.005,
        "training_data": {"modeling_period_duration": 30},
        "sagemaker_metadata": {
            "local_directory": wallets_config['training_data']['local_directory'],
            "date_suffix": "concat"
        }
    }

    model_type = modeling_config["training"]["model_type"]
    # Initialize prediction containers
    y_pred = y_test_pred.copy()
    y_pred_proba = None
    y_val_pred_binary = None
    y_val_pred_proba = None

    if model_type == "classification":
        threshold = modeling_config["predicting"]["y_pred_threshold"]
        wime_modeling_config["y_pred_threshold"] = threshold
        # Store raw probabilities then binarize
        y_pred_proba = y_test_pred.copy()
        y_pred = (y_test_pred > threshold).astype(int)
        if validation_provided:
            y_val_pred_proba = y_val_pred.copy()
            y_val_pred_binary = (y_val_pred > threshold).astype(int)

    # Remove "y_val" (raw DataFrame), use only "y_validation"
    # Update key: "y_val_pred" -> "y_validation_pred"
    if model_type == "classification":
        y_validation_pred = y_val_pred_binary
    else:
        y_validation_pred = y_val_pred if validation_provided else None

    # Build aligned continuous returns for validation (for return-vs-score plots)
    validation_target_vars_df = None
    if validation_provided:
        base_dir = Path(wallets_config['training_data']['local_s3_root'])
        concat_dir = (base_dir / 's3_uploads' / 'wallet_training_data_concatenated' /
                      wallets_config['training_data']['local_directory'])

        # Load full (unprocessed) validation target variables
        full_val_y_path = concat_dir / 'val_y.csv'
        if not full_val_y_path.exists():
            raise FileNotFoundError(f"Missing raw validation targets at {full_val_y_path}")
        full_val_y = pd.read_csv(full_val_y_path)
        if cont_target_var not in full_val_y.columns:
            raise ConfigError(
                f"Expected returns column '{cont_target_var}' not found in {full_val_y_path}. "
                f"Available columns: {full_val_y.columns.tolist()}"
            )
        returns_col = cont_target_var

        # Reuse masks from the first pass to align raw validation returns:
        # 1) epoch_mask_val reduces to the shifted offsets
        # 2) row_mask_val reduces to rows kept by feature filters
        y_epoch_returns = full_val_y.loc[epoch_mask_val, returns_col].reset_index(drop=True)
        returns_series_val = y_epoch_returns[row_mask_val].reset_index(drop=True).rename(cont_target_var)

        # Final aligned DataFrame
        validation_target_vars_df = returns_series_val.to_frame(name=cont_target_var)

    wallet_model_results = {
        "model_id": model_uri,
        "modeling_config": wime_modeling_config,
        "model_type": model_type,
        "X_train": train_X,
        "y_train": train_y,
        "X_test": X_test_final,
        "y_test": y_test_series,
        "y_pred": y_pred,
        "training_cohort_pred": None,
        "training_cohort_actuals": None,
        "X_validation": X_val_final,
        "y_validation": y_val_series,
        "validation_target_vars_df": validation_target_vars_df,
        "y_validation_pred": y_validation_pred,
        "pipeline": create_mock_pipeline(model_type)
    }

    if model_type == "classification":
        wallet_model_results["y_pred_proba"] = y_pred_proba
        # Change key from "y_val_pred_proba" to "y_validation_pred_proba"
        wallet_model_results["y_validation_pred_proba"] = y_val_pred_proba

    # Return the appropriate evaluator
    if model_type == "regression":
        return wime.RegressorEvaluator(wallet_model_results)
    else:
        return wime.ClassifierEvaluator(wallet_model_results)

# --------------------------
#      Helper Functions
# --------------------------

def _process_concatenated_split(
    split: str,
    wallets_config: dict,
    modeling_config: dict,
    data_path: Path,
    metadata: dict,
    y_pred: pd.Series,
    y_df: pd.DataFrame,
    epoch_shift: int
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Common processing for 'test' and 'val':
    1) Epoch-offset selection via select_shifted_offsets
    2) Custom X filtering via apply_row_filters
    3) Align y and y_pred with masks; labels already preprocessed upstream.
    Returns:
    - y_final (Series)
    - X_filtered (DataFrame)
    - y_pred_filtered (Series)
    - row_mask (np.ndarray): mask after feature filtering
    - epoch_mask_full (np.ndarray): mask after epoch-offset selection
    """
    # Load raw features (first column is offset_date; no headers)
    gz_path = data_path / f"{split}.csv.gz"
    csv_path = data_path / f"{split}.csv"
    x_path = gz_path if gz_path.exists() else csv_path
    X_full = pd.read_csv(x_path, header=None, compression='infer')

    # Sanity checks
    if len(X_full) != len(y_df):
        raise ValueError(f"{split}: features rows ({len(X_full)}) must match y rows ({len(y_df)})")
    if len(X_full) != len(y_pred):
        raise ValueError(f"{split}: features rows ({len(X_full)}) must match y_pred rows ({len(y_pred)})")

    # 1) Epoch-offset filtering
    X_epoch, y_epoch = select_shifted_offsets(
        X_full, y_df, wallets_config, epoch_shift, split
    )

    # Build epoch mask directly from configured target_offset_days after shift
    # (do NOT derive from X_epoch, which no longer contains the offset column).
    target_offset_days = identify_offset_ints(wallets_config, shift=epoch_shift)[f'{split}_offsets']
    epoch_mask_full = X_full.iloc[:, 0].isin(target_offset_days).to_numpy()

    # Apply the epoch mask to predictions
    y_pred_epoch = y_pred[epoch_mask_full].reset_index(drop=True)

    # Sanity: the mask rows should match rows kept by select_shifted_offsets
    if int(epoch_mask_full.sum()) != len(X_epoch):
        logger.warning(
            "%s: epoch mask mismatch (mask rows=%s vs X_epoch=%s); using config-derived mask.",
            split, int(epoch_mask_full.sum()), len(X_epoch)
        )

    # 2) Custom feature filtering
    try:
        X_filtered, row_mask = apply_row_filters(X_epoch, metadata['feature_columns'], modeling_config)
    except ValueError as e:
        # custom_transforms.apply_row_filters may raise when all rows are removed
        if "All rows filtered out by custom filters" in str(e):
            raise SkipEpochEvaluation("No rows remaining after custom feature filters.") from e
        raise
    # Defensive: if filtering returns an empty frame without raising, skip this epoch
    if len(X_filtered) == 0:
        raise SkipEpochEvaluation("No rows remaining after custom feature filters.")

    # 3) Align y and preds with the filter mask
    y_filtered = y_epoch[row_mask].reset_index(drop=True)
    y_pred_filtered = y_pred_epoch[row_mask].reset_index(drop=True)

    # y was already processed by load_concatenated_y(); just rename for clarity
    target_var = modeling_config['target']['target_var']
    y_final = y_filtered.iloc[:, 0].rename(target_var)

    if not (len(X_filtered) == len(y_final) == len(y_pred_filtered)):
        raise ValueError(
            f"{split}: post-filter length mismatch "
            f"(X={len(X_filtered)}, y={len(y_final)}, y_pred={len(y_pred_filtered)})"
        )

    return y_final, X_filtered, y_pred_filtered, row_mask, epoch_mask_full



def load_endpoint_sagemaker_predictions(
    data_type: str,
    wallets_config: dict,
    modeling_config: dict,
    date_suffix: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Load SageMaker predictions made using a SageMaker Endpoint API.

    Params:
    - data_type (str): Either 'test' or 'val'
    - wallets_config (dict): Configuration for training data paths
    - modeling_config (dict): Configuration for model parameters
    - date_suffix (str): Date suffix for file naming

    Returns:
    - tuple: (predictions_series, actuals_series) with aligned indices
    """
    # Load predictions
    pred_path = (
        Path(modeling_config['metaparams']['endpoint_preds_dir']) /
        f"endpoint_y_pred_{data_type}_{wallets_config['training_data']['local_directory']}"
        f"_{date_suffix}.csv")
    pred_df = pd.read_csv(pred_path)

    if 'score' not in pred_df.columns:
        raise ValueError(f"SageMaker predictions are missing the 'score' column. "
                        f"Available columns: {pred_df.columns}")

    pred_series = pred_df['score']

    # Check for NaN values
    if pred_series.isna().any():
        nan_count = pred_series.isna().sum()
        raise ValueError(f"Found {nan_count} NaN values in {data_type} predictions.")

    return pred_series


def load_bt_sagemaker_predictions(
    data_type: str,
    wallets_config: dict,
    date_suffix: str
) -> pd.Series:
    """
    Load SageMaker predictions made using SageMaker Batch Transform.

    Params:
    - data_type (str): Either 'test' or 'val'
    - wallets_config (dict): Configuration for training data paths
    - date_suffix (str): Date suffix for file naming

    Returns:
    - predictions_series (Series): Raw predictions without index alignment
    """
    # Load predictions
    pred_path = (
        Path(f"{wallets_config['training_data']['local_s3_root']}")
        / "s3_downloads"
        / "wallet_predictions"
        / f"{wallets_config['training_data']['download_directory']}"
        / f"{date_suffix}"
        / f"{data_type}.csv.out"
    )
    pred_df = pd.read_csv(pred_path, header=None)
    pred_series = pred_df[0]

    # Check for NaN values
    if pred_series.isna().any():
        nan_count = pred_series.isna().sum()
        raise ValueError(f"Found {nan_count} NaN values in {data_type} predictions.")

    return pred_series

def load_concatenated_y(
    data_type: str,
    wallets_config: dict,
    modeling_config: dict
) -> pd.Series:
    """
    Load concatenated target variable series for concatenated datasets.

    Params:
    - data_type (str): Name of data set type, e.g., 'test' or 'val'.
    - wallets_config (dict): Configuration for training data paths.
    - modeling_config (dict): Configuration for modeling labels.

    Returns:
    - target_series (pd.Series): Series of target values loaded from CSV.
    """
    base_dir = Path(wallets_config['training_data']['local_s3_root'])
    concat_dir = base_dir / "s3_uploads" / "wallet_training_data_concatenated"
    local_dir = f"{wallets_config['training_data']['local_directory']}"
    data_path = concat_dir / local_dir
    csv_path = data_path / f"{data_type}_y.csv"

    # Handle full y df from custom transform pipeline
    full_y_df = pd.read_csv(csv_path)
    target_series = pd.Series(preprocess_custom_labels(full_y_df, modeling_config))
    target_df = pd.DataFrame(target_series)

    # Check for NaN values
    if target_series.isna().any():
        nan_count = target_series.isna().sum()
        raise ValueError(f"Found {nan_count} NaN values in {data_type} target data from {csv_path}")

    return target_df

@u.timing_decorator()
def _apply_custom_transforms_to_concatenated_data(
    wallets_config: dict,
    modeling_config: dict,
    y_test_pred: pd.Series,
    y_test: pd.DataFrame,
    y_val_pred: pd.Series = None,
    y_val: pd.DataFrame = None,
    epoch_shift: int = 0
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame,
           pd.Series, np.ndarray, np.ndarray]:
    """
    Apply epoch-offset selection and custom feature filters to concatenated data for BOTH
    'test' and 'val'. Validation data must be provided.

    Returns:
    - y_test_final (Series)
    - X_test_final (DataFrame)
    - y_test_pred_final (Series)
    - y_val_final (Series)
    - X_val_final (DataFrame)
    - y_val_pred_final (Series)
    - row_mask_val (np.ndarray): mask after feature filtering for val
    - epoch_mask_val (np.ndarray): mask after epoch-offset selection for val
    """
    # Resolve paths
    base_dir = Path(wallets_config["training_data"]["local_s3_root"])
    concat_dir = base_dir / "s3_uploads" / "wallet_training_data_concatenated"
    local_dir = wallets_config["training_data"]["local_directory"]
    data_path = concat_dir / local_dir

    # Load metadata for transforms
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Require validation inputs so both splits follow the identical path
    if y_val is None or y_val_pred is None:
        raise ValueError("Validation data (y_val and y_val_pred) must be provided.")

    # Process splits identically
    y_test_final, X_test_final, y_test_pred_final, _, _ = _process_concatenated_split(
        "test", wallets_config, modeling_config, data_path, metadata,
        y_test_pred, y_test, epoch_shift
    )

    y_val_final, X_val_final, y_val_pred_final, row_mask_val, epoch_mask_val = _process_concatenated_split(
        "val", wallets_config, modeling_config, data_path, metadata,
        y_val_pred, y_val, epoch_shift
    )

    return (y_test_final, X_test_final, y_test_pred_final, y_val_final, X_val_final, y_val_pred_final,
            row_mask_val, epoch_mask_val)


# --------------------------
#      Utility Functions
# --------------------------

def assign_index_to_pred(
        pred_series: pd.Series,
        actuals_df: pd.DataFrame
    ) -> pd.Series:
    """
    Convert prediction series to Series with validated index alignment from actuals.

    Params:
    - pred_series (Series): Predictions from model
    - actuals_df (DataFrame): Corresponding actuals DataFrame with target index

    Returns:
    - pred_series (Series): Predictions with aligned index
    """
    # Validate input types
    if not isinstance(pred_series, pd.Series):
        raise TypeError(f"Expected Series for pred_series, got {type(pred_series)}")

    if not isinstance(actuals_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for actuals_df, got {type(actuals_df)}")

    # Validate single column
    if len(actuals_df.columns) != 1:
        raise ValueError(f"DataFrame must have exactly 1 column, found "
                         f"{len(actuals_df.columns)}: {actuals_df.columns.tolist()}")

    # Extract the series from DataFrame
    actuals_series = actuals_df.iloc[:, 0]

    # Validate lengths
    if len(pred_series) != len(actuals_series):
        raise ValueError(f"Length of y_pred ({len(pred_series)}) does "
                        f"not match length of y_true ({len(actuals_series)}).")

    # Create new series with aligned index using raw values
    aligned_pred_series = pd.Series(pred_series.values, index=actuals_series.index)

    # Check for NaN values
    if aligned_pred_series.isna().any():
        nan_count = aligned_pred_series.isna().sum()
        raise ValueError(f"Found {nan_count} NaN values in predictions.")

    return aligned_pred_series


def create_mock_pipeline(model_type):
    """
    Create a mock pipeline for wime evaluation compatibility.

    Params:
    - objective (str): XGBoost objective parameter

    Returns:
    - Mock pipeline object with required methods
    """
    return type('MockPipeline', (), {
        'named_steps': {'estimator': type('MockModel', (), {
            f'get_params': lambda self: {'objective': f'{model_type}'}
        })()},
        '__getitem__': lambda self, key: type('MockTransformer', (), {
            'transform': lambda self, X: X
        })()
    })()
