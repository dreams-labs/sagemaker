import sys
from pathlib import Path
import pandas as pd
from typing import Tuple

# Add wallet_insights to path    # pylint:disable=wrong-import-position
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
import wallet_insights.model_evaluation as wime


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


def create_mock_pipeline(objective: str):
    """
    Create a mock pipeline for SageMaker evaluation compatibility.

    Params:
    - objective (str): XGBoost objective parameter

    Returns:
    - Mock pipeline object with required methods
    """
    return type('MockPipeline', (), {
        'named_steps': {'estimator': type('MockModel', (), {
            'get_params': lambda self: {'objective': objective}
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

    # Identify target variable and model type
    target_variable = y_val_true_series.name or y_train.columns[0]
    objective = sage_wallets_modeling_config['training']['hyperparameters']['objective']
    model_type = 'regression' if objective[:3] == 'reg' else 'unknown'

    # Create model_id and modeling_config
    model_id = f"sagemaker_{sage_wallets_config['training_data']['local_directory']}_{date_suffix}"

    modeling_config = {
        'target_variable': target_variable,
        'model_type': model_type,
        'returns_winsorization': 0.005,
        'training_data': {
            'modeling_period_duration': 30
        },
        'sagemaker_metadata': {
            'objective': objective,
            'local_directory': sage_wallets_config['training_data']['local_directory'],
            'date_suffix': date_suffix
        }
    }

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
        'pipeline': create_mock_pipeline(objective)
    }

    # Create and return evaluator
    wallet_evaluator = wime.RegressorEvaluator(wallet_model_results)
    return wallet_evaluator


def run_sagemaker_evaluation(
    sage_wallets_config: dict,
    sage_wallets_modeling_config: dict,
    date_suffix: str
) -> wime.RegressorEvaluator:
    """
    Complete SageMaker evaluation pipeline: load data, create evaluator, run reports.

    Params:
    - sage_wallets_config (dict): Configuration for training data paths
    - sage_wallets_modeling_config (dict): Configuration for model parameters
    - date_suffix (str): Date suffix for file naming

    Returns:
    - RegressorEvaluator: Evaluator after running summary report and plots
    """
    wallet_evaluator = create_sagemaker_evaluator(
        sage_wallets_config, sage_wallets_modeling_config, date_suffix
    )

    # Run evaluation
    wallet_evaluator.summary_report()
    wallet_evaluator.plot_wallet_evaluation()

    return wallet_evaluator