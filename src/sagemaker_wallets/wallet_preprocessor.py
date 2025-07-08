"""
Preprocesses training data for SageMaker XGBoost compatibility.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SageWalletsPreprocessor:
    """
    Handles preprocessing of training data splits for SageMaker XGBoost requirements.

    SageMaker XGBoost expects:
    - Target variable as first column
    - No column headers
    - No missing values
    - All numeric data types
    """

    def __init__(self):
        pass

    def preprocess_training_data(self, training_data: dict) -> dict:
        """
        Preprocess all training data splits for SageMaker compatibility.

        Params:
        - training_data (dict): Raw training data with keys like 'x_train', 'y_train', etc.

        Returns:
        - dict: Preprocessed training data ready for SageMaker upload
        """
        # Split names that need X preprocessing
        x_splits = ['x_train', 'x_test', 'x_eval', 'x_val']
        y_splits = ['y_train', 'y_test', 'y_eval', 'y_val']

        processed_data = {}

        logger.info("Starting preprocessing for SageMaker XGBoost compatibility...")

        for x_split, y_split in zip(x_splits, y_splits):
            if x_split not in training_data or y_split not in training_data:
                raise ValueError(f"Missing required splits: {x_split}, {y_split}")

            # Extract base split name (train, test, eval, val)
            split_name = x_split.replace('x_', '')

            # Validate index alignment first
            self._validate_index_alignment(training_data[x_split], training_data[y_split], split_name)

            # Preprocess X data
            x_processed = self._preprocess_x_data(training_data[x_split], x_split)

            # Combine X and y with target as first column
            combined_data = self._combine_x_y_data(x_processed, training_data[y_split])

            # Store combined data (no separate X/y anymore)
            processed_data[split_name] = combined_data

            logger.info(f"Preprocessed {split_name}: {combined_data.shape[0]:,} rows × {combined_data.shape[1]} cols.")

        return processed_data

    def _preprocess_x_data(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Preprocess X data for SageMaker XGBoost compatibility.

        Params:
        - df (DataFrame): Input X data
        - split_name (str): Name of split for error reporting

        Returns:
        - DataFrame: Preprocessed data ready for SageMaker
        """
        # Handle missing values
        if df.isnull().any().any():
            logger.warning(f"Found NaN values in {split_name}, filling with 0.")
            df = df.fillna(0)

        # Ensure numeric data types
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Non-numeric columns found in {split_name}: {list(non_numeric_cols)}")

        # Convert all to float32 for consistency
        df = df.astype('float32')

        return df

    def _validate_index_alignment(self, x_df: pd.DataFrame, y_df: pd.DataFrame, split_name: str):
        """
        Validate that X and y DataFrames have matching indices.

        Params:
        - x_df (DataFrame): Features DataFrame
        - y_df (DataFrame): Target DataFrame
        - split_name (str): Name of split for error reporting
        """
        if x_df.empty or y_df.empty:
            raise ValueError(f"Empty DataFrame found for {split_name}")

        if not np.array_equal(x_df.index, y_df.index):
            raise ValueError(f"Index mismatch between X and y for {split_name}.")

    def _combine_x_y_data(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine X and y DataFrames with target variable as first column.

        Params:
        - x_df (DataFrame): Preprocessed features
        - y_df (DataFrame): Target variable

        Returns:
        - DataFrame: Combined data with target as first column, no headers
        """
        # Combine with target as first column
        combined_df = pd.concat([y_df, x_df], axis=1)

        return combined_df
