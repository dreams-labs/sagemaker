import sys
import logging
from pathlib import Path
from typing import Tuple,Union
import pandas as pd

# Add wallet_insights to path    # pylint:disable=wrong-import-position
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
from sage_wallet_modeling.wallet_preprocessor import SageWalletsPreprocessor
from script_modeling.custom_transforms import preprocess_custom_labels
import wallet_insights.model_evaluation as wime
from utils import ConfigError


# pylint:disable=invalid-name  # X isn't lowercase
# Set up logger at the module level
logger = logging.getLogger(__name__)


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

        # Convert to binary for classification models
        preprocessor = SageWalletsPreprocessor(wallets_config, modeling_config)
        y_train = preprocessor.preprocess_y_data(y_train, 'train')
        y_test = preprocessor.preprocess_y_data(y_test, 'test')
        y_pred_proba = y_test_pred
        y_val = preprocessor.preprocess_y_data(y_val, 'val')
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
    y_test_pred: pd.Series,
    y_test: pd.DataFrame,
    y_val_pred: pd.Series = None,
    y_val: pd.DataFrame = None
) -> Union[wime.RegressorEvaluator, wime.ClassifierEvaluator]:
    """
    Create a SageMaker evaluator for concatenated model results with optional validation set.

    Params:
    - wallets_config (dict): Configuration for training data paths
    - modeling_config (dict): Configuration for model parameters
    - y_test_pred (Series): Predicted values for the concatenated test set
    - y_test (DataFrame): Single-column actual target for the concatenated test set
    - y_val_pred (Series, optional): Predicted values for validation set
    - y_val (DataFrame, optional): Single-column actual target for the validation set

    Returns:
    - RegressorEvaluator or ClassifierEvaluator: Configured evaluator
    """
    # Validate test target
    if not isinstance(y_test, pd.DataFrame) or y_test.shape[1] != 1:
        raise ValueError("Expected y_test to be a single-column DataFrame")
    target_var = y_test.columns[0]
    y_test_series = y_test.iloc[:, 0].rename(target_var)
    y_test_pred.index = y_test_series.index

    # Validate optional validation target
    validation_provided = y_val is not None and y_val_pred is not None
    if validation_provided:
        if not isinstance(y_val, pd.DataFrame) or y_val.shape[1] != 1:
            raise ValueError("Expected y_val to be a single-column DataFrame")
        y_val_series = y_val.iloc[:, 0].rename(target_var)
        y_val_pred.index = y_val_series.index

    # Load concatenated training and test features
    base_dir = Path(wallets_config["training_data"]["local_s3_root"])
    concat_dir = base_dir / "s3_uploads" / "wallet_training_data_concatenated"
    local_dir = wallets_config["training_data"]["local_directory"]
    if wallets_config["training_data"].get("dataset", "dev") == "dev":
        local_dir = f"{local_dir}_dev"
    data_path = concat_dir / local_dir

    train_df = pd.read_csv(data_path / "train.csv", header=None)
    test_df = pd.read_csv(data_path / "test.csv", header=None)

    y_train = train_df.iloc[:, 0]
    if modeling_config['target']['custom_y']:
        # No y appended
        X_train = train_df
    else:
        # Appended y in first column
        X_train = train_df.iloc[:, 1:]
    X_test = test_df

    # Ensure feature count matches
    expected_n = X_train.shape[1]
    if X_test.shape[1] != expected_n:
        raise ValueError(
            f"Feature count mismatch: train has {expected_n}, test has {X_test.shape[1]}"
        )

    # Rename feature columns to avoid numeric-only names
    X_train.columns = [f"feature_{i}" for i in range(expected_n)]
    X_test.columns = [f"feature_{i}" for i in range(expected_n)]

    # Prepare validation placeholders
    X_validation = None
    y_validation = None
    if validation_provided:
        X_validation = None
        y_validation = y_val_series

    # Build modeling config for evaluator
    wime_modeling_config = {
        "target_variable": target_var,
        "model_type": modeling_config["training"]["model_type"],
        "returns_winsorization": 0.005,
        "training_data": {"modeling_period_duration": 30},
        "sagemaker_metadata": {
            "local_directory": local_dir,
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

    model_id = f"sagemaker_concat_{local_dir}"

    wallet_model_results = {
        "model_id": model_id,
        "modeling_config": wime_modeling_config,
        "model_type": model_type,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test_series,
        "y_pred": y_pred,
        "training_cohort_pred": None,
        "training_cohort_actuals": None,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "validation_target_vars_df": y_val,
        # Remove "y_val" (raw DataFrame), use only "y_validation"
        # Update key: "y_val_pred" -> "y_validation_pred"
        "y_validation_pred": y_val_pred_binary if model_type == "classification" else (y_val_pred if validation_provided else None),
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
    local_dir = wallets_config['training_data']['local_directory']
    data_path = concat_dir / local_dir
    csv_path = data_path / f"{data_type}_y.csv"

    # Handle full y df from custom transform pipeline
    if modeling_config['target']['custom_y']:
        full_y_df = pd.read_csv(csv_path)
        target_series = pd.Series(preprocess_custom_labels(full_y_df, modeling_config))
        target_df = pd.DataFrame(target_series)
        # target_df = target_df[[modeling_config['target']['target_var']]]
        # target_series = target_df[modeling_config['target']['target_var']]

    # Handle single column y df from base pipeline
    else:
        # Load the target CSV: test_y.csv or val_y.csv depending on data_type
        target_df = pd.read_csv(csv_path,header=None)
        target_series = pd.Series(target_df.iloc[:, 0].values)


    # Check for NaN values
    if target_series.isna().any():
        nan_count = target_series.isna().sum()
        raise ValueError(f"Found {nan_count} NaN values in {data_type} target data from {csv_path}")

    return target_df


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
