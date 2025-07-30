import sys
from pathlib import Path
from typing import Tuple
import pandas as pd

# Add wallet_insights to path    # pylint:disable=wrong-import-position
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
import wallet_insights.model_evaluation as wime
from utils import ConfigError


# pylint:disable=invalid-name  # X isn't lowercase

def load_sagemaker_predictions(
    data_type: str,
    sage_wallets_config: dict,
    sage_wallets_modeling_config: dict,
    date_suffix: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Load SageMaker predictions and corresponding actuals for a given data type.

    Params:
    - data_type (str): Either 'test' or 'val'
    - sage_wallets_config (dict): Configuration for training data paths
    - sage_wallets_modeling_config (dict): Configuration for model parameters
    - date_suffix (str): Date suffix for file naming

    Returns:
    - tuple: (predictions_series, actuals_series) with aligned indices
    """
    # Load predictions
    pred_path = (
        Path(sage_wallets_modeling_config['metaparams']['endpoint_preds_dir']) /
        f"endpoint_y_pred_{data_type}_{sage_wallets_config['training_data']['local_directory']}"
        f"_{date_suffix}.csv")
    pred_df = pd.read_csv(pred_path)

    if 'score' not in pred_df.columns:
        raise ValueError(f"SageMaker predictions are missing the 'score' column. "
                        f"Available columns: {pred_df.columns}")
    pred_series = pred_df['score']

    # Load actuals
    training_data_path = (
        Path(f"../s3_uploads") / "wallet_training_data_queue" /
        f"{sage_wallets_config['training_data']['local_directory']}"
    )
    actuals_path = training_data_path / f"y_{data_type}_{date_suffix}.parquet"
    actuals_df = pd.read_parquet(actuals_path)

    if len(actuals_df.columns) > 1:
        raise ValueError(f"Found unexpected columns in y_{data_type}_df. "
                        f"Expected 1 column, found {actuals_df.columns}.")
    actuals_series = actuals_df.iloc[:, 0]

    # Validate lengths and align indices
    if len(pred_series) != len(actuals_series):
        raise ValueError(f"Length of y_{data_type}_pred ({len(pred_series)}) does "
                        f"not match length of y_{data_type}_true ({len(actuals_series)}).")

    pred_series.index = actuals_series.index

    return pred_series, actuals_series


def create_mock_pipeline():
    """
    Create a mock pipeline for SageMaker evaluation compatibility.

    Params:
    - objective (str): XGBoost objective parameter

    Returns:
    - Mock pipeline object with required methods
    """
    return type('MockPipeline', (), {
        'named_steps': {'estimator': type('MockModel', (), {
            'get_params': lambda self: {'objective': 'mock_objective'}
        })()},
        '__getitem__': lambda self, key: type('MockTransformer', (), {
            'transform': lambda self, X: X
        })()
    })()


def create_sagemaker_evaluator(
    sage_wallets_config: dict,
    sage_wallets_modeling_config: dict,
    date_suffix: str
) -> wime.RegressorEvaluator:
    """
    Create a complete SageMaker wallet evaluator with all required data loaded.

    Params:
    - sage_wallets_config (dict): Configuration for training data paths
    - sage_wallets_modeling_config (dict): Configuration for model parameters
    - date_suffix (str): Date suffix for file naming

    Returns:
    - RegressorEvaluator: Configured evaluator ready for analysis
    """
    # Load predictions and actuals
    y_test_pred_series, y_test_true_series = load_sagemaker_predictions(
        'test', sage_wallets_config, sage_wallets_modeling_config, date_suffix
    )
    y_val_pred_series, y_val_true_series = load_sagemaker_predictions(
        'val', sage_wallets_config, sage_wallets_modeling_config, date_suffix
    )

    # Load remaining training data
    training_data_path = (
        Path(f"../s3_uploads") / "wallet_training_data_queue" /
        f"{sage_wallets_config['training_data']['local_directory']}"
    )
    X_train = pd.read_parquet(training_data_path / f"x_train_{date_suffix}.parquet")
    y_train = pd.read_parquet(training_data_path / f"y_train_{date_suffix}.parquet")
    X_test = pd.read_parquet(training_data_path / f"x_test_{date_suffix}.parquet")
    X_val = pd.read_parquet(training_data_path / f"x_val_{date_suffix}.parquet")
    y_val = pd.read_parquet(training_data_path / f"y_val_{date_suffix}.parquet")

    # Create model_id and modeling_config
    model_id = f"sagemaker_{sage_wallets_config['training_data']['local_directory']}_{date_suffix}"

    target_variable = y_val_true_series.name or y_train.columns[0]
    model_type = sage_wallets_modeling_config['training']['model_type']
    modeling_config = {
        'target_variable': target_variable,
        'model_type': model_type,
        'returns_winsorization': 0.005,
        'training_data': {
            'modeling_period_duration': 30
        },
        'sagemaker_metadata': {
            'local_directory': sage_wallets_config['training_data']['local_directory'],
            'date_suffix': date_suffix
        }
    }

    # Include y_pred_threshold for classification models
    if model_type == 'classification':
        y_pred_thresh = sage_wallets_modeling_config['target']['classification']['threshold']
        modeling_config['y_pred_threshold'] = y_pred_thresh

    # Create wallet_model_results dictionary
    wallet_model_results = {
        'model_id': model_id,
        'modeling_config': modeling_config,
        'model_type': model_type,

        # Training data
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test_true_series,
        'y_pred': y_test_pred_series,
        'training_cohort_pred': None,
        'training_cohort_actuals': None,

        # Validation data
        'X_validation': X_val,
        'y_validation': y_val_true_series,
        'y_validation_pred': y_val_pred_series,
        'validation_target_vars_df': y_val,

        # Mock pipeline
        'pipeline': create_mock_pipeline()
    }

    # Create and return evaluator
    if model_type == 'regression':
        wallet_evaluator = wime.RegressorEvaluator(wallet_model_results)
    elif model_type == 'classification':
        wallet_evaluator = wime.ClassifierEvaluator(wallet_model_results)
    else:
        raise ConfigError(f"Unknown model type {model_type} found in config.")

    return wallet_evaluator


def run_sagemaker_evaluation(
    sage_wallets_config: dict,
    sage_wallets_modeling_config: dict,
    date_suffix: str
) -> wime.RegressorEvaluator:


# --------------------------
#      Utility Functions
# --------------------------

def assign_index_to_pred(
        pred_series: pd.Series,
        actuals_df: pd.DataFrame
    ) -> pd.Series:
    """
    Complete SageMaker evaluation pipeline: load data, create evaluator, run reports.
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
        raise ValueError(f"DataFrame must have exactly 1 column, found {len(actuals_df.columns)}: {actuals_df.columns.tolist()}")

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



def create_mock_pipeline():
    """
    Create a mock pipeline for SageMaker evaluation compatibility.

    Params:
    - objective (str): XGBoost objective parameter

    Returns:
    - Mock pipeline object with required methods
    """
    return type('MockPipeline', (), {
        'named_steps': {'estimator': type('MockModel', (), {
            'get_params': lambda self: {'objective': 'mock_objective'}
        })()},
        '__getitem__': lambda self, key: type('MockTransformer', (), {
            'transform': lambda self, X: X
        })()
    })()
