import sys
import logging
from typing import Tuple, Union, Dict
from pathlib import Path
import json
import pandas as pd


# pylint:disable=wrong-import-position
# ensure script_modeling directory is on path for its module
script_modeling_dir = Path(__file__).resolve().parents[1] / "script_modeling"
sys.path.insert(0, str(script_modeling_dir))
from custom_transforms import apply_custom_feature_filters, preprocess_custom_labels, select_shifted_offsets

# Import from data-science repo
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
import wallet_insights.model_evaluation as wime
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

def create_sagemaker_evaluator(
    wallets_config: dict,
    modeling_config: dict,
    date_suffix: str,
    y_test_pred: pd.Series,
    y_val_pred: pd.Series
 ) -> Union[wime.RegressorEvaluator,wime.ClassifierEvaluator]:
    """
    Create a complete SageMaker wallet evaluator with all required data loaded.

    Params:
    - wallets_config (dict): Configuration for training data paths
    - modeling_config (dict): Configuration for model parameters
    - date_suffix (str): Date suffix for file naming
    - y_test_pred (pd.Series): Predicted values for the test set
    - y_test_val (pd.Series): Predicted values for the validation set

    Returns:
    - RegressorEvaluator: Configured evaluator ready for analysis
    """
    # 1. Load and Prepare Training Data
    # ---------------------------------
    model_type = modeling_config['training']['model_type']

    # Load remaining training data
    training_data_path = (
        Path(f"{wallets_config['training_data']['local_s3_root']}")
        / "s3_uploads"
        / "wallet_training_data_queue"
        / f"{wallets_config['training_data']['training_data_directory']}"
    )
    X_train = pd.read_parquet(training_data_path / f"x_train_{date_suffix}.parquet")
    y_train = pd.read_parquet(training_data_path / f"y_train_{date_suffix}.parquet")
    X_test = pd.read_parquet(training_data_path / f"x_test_{date_suffix}.parquet")
    y_test = pd.read_parquet(training_data_path / f"y_test_{date_suffix}.parquet")
    X_val = pd.read_parquet(training_data_path / f"x_val_{date_suffix}.parquet")
    y_val = pd.read_parquet(training_data_path / f"y_val_{date_suffix}.parquet")

    # Assign indices to pred arrays
    y_test_pred = assign_index_to_pred(y_test_pred,y_test)
    y_val_pred = assign_index_to_pred(y_val_pred,y_val)


    # 2. Prepare wime modeling config
    # -------------------------------
    # Create modeling_config in format expected by wallet_insights.model_evaluation
    wime_modeling_config = {
        'target_variable': y_train.columns[0],
        'model_type': model_type,
        'returns_winsorization': 0.005,
        'training_data': {
            'modeling_period_duration': 30
        },
        'sagemaker_metadata': {
            'local_directory': wallets_config['training_data']['local_directory'],
            'date_suffix': date_suffix
        }
    }

    # 3. Handle classification vars
    # -----------------------------
    if model_type == 'classification':
        # Add y_pred_theshold to config
        y_pred_thresh = modeling_config['predicting']['y_pred_threshold']
        wime_modeling_config['y_pred_threshold'] = y_pred_thresh

        y_pred_proba = y_test_pred
        y_val_pred_proba = y_val_pred

        # Apply y_pred_threshold to convert probabilities to binary predictions
        y_test_pred = (y_test_pred > y_pred_thresh).astype(int)
        y_val_pred = (y_val_pred > y_pred_thresh).astype(int)

    # 4. Prepare model results
    # ------------------------
    # Create model_id
    model_id = f"sagemaker_{wallets_config['training_data']['local_directory']}_{date_suffix}"

    wallet_model_results = {
        'model_id': model_id,
        'modeling_config': wime_modeling_config,
        'model_type': model_type,

        # Training data
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.iloc[:, 0],
        'y_test': y_test.iloc[:, 0],
        'y_pred': y_test_pred,
        'training_cohort_pred': None,
        'training_cohort_actuals': None,

        # Validation data
        'X_validation': X_val,
        'y_validation': y_val.iloc[:, 0],
        'y_val_pred': y_val_pred,
        'y_val': y_val,

        # Mock pipeline
        'pipeline': create_mock_pipeline(model_type)
    }

    if model_type == 'classification':
        wallet_model_results['y_pred_proba'] = y_pred_proba
        wallet_model_results['y_val_pred_proba'] = y_val_pred_proba

    # 5. Create and return evaluator
    # ------------------------------
    if model_type == 'regression':
        wallet_evaluator = wime.RegressorEvaluator(wallet_model_results)
    elif model_type == 'classification':
        wallet_evaluator = wime.ClassifierEvaluator(wallet_model_results)
    else:
        raise ConfigError(f"Unknown model type {model_type} found in config.")

    return wallet_evaluator


def create_concatenated_sagemaker_evaluator(
    wallets_config: dict,
    modeling_config: dict,
    model_uri: str,
    y_test_pred: pd.Series,
    y_test: pd.DataFrame,
    y_val_pred: pd.Series = None,
    y_val: pd.DataFrame = None,
    epoch_shift: int = 0
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

    Returns:
    - RegressorEvaluator or ClassifierEvaluator: Configured evaluator
      (May return None if all rows are filtered out for the given epoch.)
    """
    # Apply custom transforms and get filtered data
    try:
        y_test_final, X_test_final, y_test_pred_final, y_val_final, X_val_final, y_val_pred_final = \
            _apply_custom_transforms_to_concatenated_data(
                wallets_config, modeling_config, y_test_pred, y_test, y_val_pred, y_val, epoch_shift
            )
    except SkipEpochEvaluation:
        logger.warning("Skipping evaluation for epoch_shift=%s (all rows filtered out).", epoch_shift)
        return None

    # For custom transforms, we still need to load train data normally
    y_train, X_train, _ = _load_concatenated_features(wallets_config)
    target_var = y_test_final.name
    y_test_series = y_test_final
    y_test_pred = y_test_pred_final

    # Set validation data
    validation_provided = y_val_final is not None
    y_val_series = y_val_final if validation_provided else None
    y_val_pred = y_val_pred_final if validation_provided else None

    # Build modeling config for evaluator
    wime_modeling_config = {
        "target_variable": target_var,
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

    wallet_model_results = {
        "model_id": model_uri,
        "modeling_config": wime_modeling_config,
        "model_type": model_type,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test_final,
        "y_test": y_test_series,
        "y_pred": y_pred,
        "training_cohort_pred": None,
        "training_cohort_actuals": None,
        "X_validation": X_val_final,
        "y_validation": y_val_series,
        "validation_target_vars_df": y_val,
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
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Common processing for 'test' and 'val':
    1) Epoch-offset selection via select_shifted_offsets
    2) Custom X filtering via apply_custom_feature_filters
    3) Align y and y_pred with masks; labels already preprocessed upstream.
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
    # For predictions, wrap as a DataFrame to reuse the same function
    _, y_pred_epoch_df = select_shifted_offsets(
        X_full, y_pred.to_frame("__pred__"), wallets_config, epoch_shift, split
    )
    y_pred_epoch = y_pred_epoch_df["__pred__"].reset_index(drop=True)

    # 2) Custom feature filtering
    try:
        X_filtered, row_mask = apply_custom_feature_filters(X_epoch, metadata, modeling_config)
    except ValueError as e:
        # custom_transforms.apply_custom_feature_filters may raise when all rows are removed
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

    return y_final, X_filtered, y_pred_filtered



# Helper to load concatenated train/test features and y_train, and rename columns
def _load_concatenated_features(
    wallets_config: Dict
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load concatenated train/test CSVs, extract y_train, ensure feature count matches,
    and rename feature columns for X_train and X_test.

    Returns:
    - y_train (Series)
    - X_train (DataFrame)
    - X_test (DataFrame)
    """
    base_dir = Path(wallets_config['training_data']['local_s3_root'])
    concat_dir = base_dir / 's3_uploads' / 'wallet_training_data_concatenated'
    local_dir = wallets_config['training_data']['local_directory']
    data_path = concat_dir / local_dir

    train_gz = data_path / 'train.csv.gz'
    test_gz = data_path / 'test.csv.gz'
    train_csv = data_path / 'train.csv'
    test_csv = data_path / 'test.csv'

    train_path = train_gz if train_gz.exists() else train_csv
    test_path = test_gz if test_gz.exists() else test_csv

    X_train = pd.read_csv(train_path, header=None, compression='infer')
    X_test = pd.read_csv(test_path, header=None, compression='infer')

    # First column is always y_train
    y_train = X_train.iloc[:, 0]

    # Validate feature dimensions
    expected_n = X_train.shape[1]
    if X_test.shape[1] != expected_n:
        raise ValueError(
            f"Feature count mismatch: train has {expected_n}, test has {X_test.shape[1]}"
        )

    # Rename feature columns
    feature_names = [f"feature_{i}" for i in range(expected_n)]
    X_train.columns = feature_names
    X_test.columns = feature_names

    return y_train, X_train, X_test


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
        / f"{wallets_config['training_data']['local_directory']}"
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


def _apply_custom_transforms_to_concatenated_data(
    wallets_config: dict,
    modeling_config: dict,
    y_test_pred: pd.Series,
    y_test: pd.DataFrame,
    y_val_pred: pd.Series = None,
    y_val: pd.DataFrame = None,
    epoch_shift: int = 0
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
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
    y_test_final, X_test_final, y_test_pred_final = _process_concatenated_split(
        "test", wallets_config, modeling_config, data_path, metadata,
        y_test_pred, y_test, epoch_shift
    )

    y_val_final, X_val_final, y_val_pred_final = _process_concatenated_split(
        "val", wallets_config, modeling_config, data_path, metadata,
        y_val_pred, y_val, epoch_shift
    )

    return y_test_final, X_test_final, y_test_pred_final, y_val_final, X_val_final, y_val_pred_final


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
