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

    def __init__(self, sage_wallets_config: dict):
        self.fill_na_config = sage_wallets_config['preprocessing']['fill_na']

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
        x_column_reference = None

        logger.info("Starting preprocessing for SageMaker XGBoost compatibility...")

        for x_split, y_split in zip(x_splits, y_splits):
            if x_split not in training_data or y_split not in training_data:
                raise ValueError(f"Missing required splits: {x_split}, {y_split}")

            # Extract base split name (train, test, eval, val)
            split_name = x_split.replace('x_', '')

            # Validate index alignment first
            self._validate_index_alignment(
                training_data[x_split],
                training_data[y_split],
                split_name
            )

            # Validate X column consistency across splits
            if x_column_reference is None:
                x_column_reference = training_data[x_split].columns
            else:
                self._validate_x_column_consistency(
                    training_data[x_split],
                    x_column_reference,
                    split_name
                )

            # Preprocess X data
            x_processed = self._preprocess_x_data(training_data[x_split], x_split)

            # Combine X and y with target as first column
            combined_data = self._combine_x_y_data(x_processed, training_data[y_split])

            # Store combined data (no separate X/y anymore)
            processed_data[split_name] = combined_data

            logger.info(f"Preprocessed {split_name}: {combined_data.shape[0]:,} rows "
                        f"× {combined_data.shape[1]} cols.")

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
        # Handle missing values with intelligent fill strategy
        df = self._handle_missing_values(df, split_name)

        # Ensure numeric data types
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Non-numeric columns found in {split_name}: "
                             f"{list(non_numeric_cols)}")

        # Convert all to float32 for consistency
        df = df.astype('float32')

        return df

    def _handle_missing_values(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Handle missing values using configured fill strategies.

        Params:
        - df (DataFrame): Input DataFrame with potential NaN values
        - split_name (str): Name of split for error reporting

        Returns:
        - DataFrame: DataFrame with NaN values filled
        """
        if not df.isnull().any().any():
            return df

        df = df.copy()
        columns_with_na = df.columns[df.isnull().any()].tolist()
        filled_columns = []

        # Apply configured fill strategies
        for col in columns_with_na:
            # Strip cw_ prefix for matching
            col_for_matching = col.replace('cw_', '', 1) if col.startswith('cw_') else col

            # Check against configured prefixes
            fill_applied = False
            for prefix, fill_value in self.fill_na_config.items():
                if col_for_matching.startswith(prefix):
                    if isinstance(fill_value, str):
                        # Apply aggregation function
                        if fill_value == 'max':
                            df[col] = df[col].fillna(df[col].max())
                        elif fill_value == 'min':
                            df[col] = df[col].fillna(df[col].min())
                        elif fill_value == 'mean':
                            df[col] = df[col].fillna(df[col].mean())
                        elif fill_value == 'median':
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            raise ValueError(f"Unknown aggregation function '{fill_value}' "
                                             f"for column {col}")
                    else:
                        # Apply numeric fill
                        df[col] = df[col].fillna(fill_value)

                    filled_columns.append(col)
                    fill_applied = True
                    break

            if not fill_applied:
                # Fill with 0 as fallback
                df[col] = df[col].fillna(0)

        # Log unexpected columns that weren't covered by config
        unexpected_columns = [col for col in columns_with_na if col not in filled_columns]
        if unexpected_columns:
            logger.warning(f"Unexpected columns with NaN values in {split_name}, "
                           f"filled with 0: {unexpected_columns}")

        logger.info(f"Filled NaN values in {len(columns_with_na)} columns for {split_name}.")

        return df

    def _validate_index_alignment(
            self,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            split_name: str
        ):
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

    def _validate_x_column_consistency(
            self,
            x_df: pd.DataFrame,
            reference_columns: pd.Index,
            split_name: str
        ):
        """
        Validate that X DataFrame has consistent columns across all splits.

        Params:
        - x_df (DataFrame): Current X DataFrame to validate
        - reference_columns (Index): Known columns from a prior X df, used to
            ensure consistency
        - split_name (str): Name of split for error reporting
        """
        # Validate minimum feature count
        if len(x_df.columns) < 2:
            raise ValueError(f"Insufficient features in {split_name}: "
                             f"{len(x_df.columns)} columns. Need at least "
                             "2 columns (target will be added as first column).")

        # Validate columns match
        if len(x_df.columns) != len(reference_columns):
            raise ValueError(f"Column count mismatch in {split_name}: "
                             f"expected {len(reference_columns)}, got {len(x_df.columns)}")

        if not x_df.columns.equals(reference_columns):
            missing_cols = reference_columns.difference(x_df.columns).tolist()
            extra_cols = x_df.columns.difference(reference_columns).tolist()

            if missing_cols or extra_cols:
                error_msg = f"Column mismatch in {split_name}:"
                if missing_cols:
                    error_msg += f" missing {missing_cols}"
                if extra_cols:
                    error_msg += f" extra {extra_cols}"
                raise ValueError(error_msg)
            else:
                # Same columns but different order - show the first few mismatches
                mismatched_positions = []
                for i, (ref_col, curr_col) in enumerate(zip(reference_columns, x_df.columns)):
                    if ref_col != curr_col:
                        mismatched_positions.append(f"pos {i}: expected '{ref_col}', "
                                                    f"got '{curr_col}'")
                    if len(mismatched_positions) >= 3:  # Limit to first 3 mismatches
                        break

                mismatch_details = "; ".join(mismatched_positions)
                if len(mismatched_positions) == 3:
                    mismatch_details += "..."

                raise ValueError(f"Column order mismatch in {split_name}: {mismatch_details}")

    def _combine_x_y_data(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine X and y DataFrames with target variable as first column.

        Params:
        - x_df (DataFrame): Preprocessed features
        - y_df (DataFrame): Target variable

        Returns:
        - DataFrame: Combined data with target as first column, no headers
        """
        if len(y_df.columns) != 1:
            raise ValueError(f"Found {len(y_df.columns)} columns in y df, which "
                             "should only have 1 column.")

        # Combine with target as first column
        combined_df = pd.concat([y_df, x_df], axis=1)
        logger.info(f"Merged y df with target var {y_df.columns[0]} with X data.")

        return combined_df
