"""
Preprocesses training data for SageMaker XGBoost compatibility.
"""
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SageWalletsPreprocessor:
    """
    Handles preprocessing of training data splits for SageMaker XGBoost requirements.

    Params:
    - sage_wallets_config (dict): from yaml

    SageMaker XGBoost expects:
    - Target variable as first column
    - No column headers
    - No missing values
    - All numeric data types
    """

    def __init__(
            self,
            sage_wallets_config: dict,
            modeling_config: dict
        ):
        # Configs
        self.wallets_config = sage_wallets_config
        self.modeling_config = modeling_config
        self.preprocessing_config = self.wallets_config['preprocessing']
        self.dataset = self.wallets_config['training_data'].get('dataset', 'dev')

        # Set up local output directory for this run
        base_dir = (Path(f"{self.wallets_config['training_data']['local_s3_root']}")
                    / "s3_uploads"
                    / "wallet_training_data_preprocessed")
        if not base_dir.exists():
            raise FileNotFoundError(f"Expected preprocessed data base directory not found: {base_dir}")
        self.output_base = base_dir / self.wallets_config["training_data"]["local_directory"]
        if self.dataset == 'dev':
            self.output_base = self.output_base.with_name(f"{self.output_base.name}_dev")
        self.output_base.mkdir(exist_ok=True)

        # Date suffix is used in saved file names only
        self.date_suffix = None


    # -------------------------
    #     Primary Interface
    # -------------------------
    def preprocess_training_data(self, training_data: dict, date_suffix: str) -> dict:
        """
        Preprocess all training data splits for SageMaker compatibility.

        Params:
        - training_data (dict): Raw training data with keys like 'x_train', 'y_train', etc.
        - date_suffix (str): Date suffix for file naming (e.g., "250301")

        Returns:
        - dict: Preprocessed training data ready for SageMaker upload
        """
        # Store date_suffix for use in file operations
        self.date_suffix = date_suffix

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
            x_preprocessed = self._preprocess_x_data(training_data[x_split], x_split)

            # Store preprocessed data in our results dict
            if split_name in ['train', 'eval']:
                # Train and Eval sets need the target var appended as first column
                y_preprocessed = self.preprocess_y_data(training_data[y_split], y_split)
                combined_data = self._combine_x_y_data(x_preprocessed, y_preprocessed)
                processed_data[split_name] = combined_data
            else:
                # Test and Val sets are only used for scoring, and shouldn't have targets
                processed_data[split_name] = x_preprocessed

            # Save preprocessed split to local file
            self._save_preprocessed_df(processed_data[split_name], split_name)

            logger.info(f"Preprocessed {split_name}: {processed_data[split_name].shape[0]:,} rows "
                        f"× {processed_data[split_name].shape[1]} cols.")

        # Save metadata alongside CSV files
        processed_data['metadata'] = self._compile_training_metadata(training_data)
        metadata_file = self.output_base / self.date_suffix / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data['metadata'], f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        logger.milestone(f"Successfully preprocessed data for '{date_suffix}'.")

        return processed_data


    def preprocess_x_df(self, x_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a standalone feature DataFrame for inference.

        Params:
        - x_df (DataFrame): Feature-only DataFrame
        - split_name (str): Used for logging and error messages

        Returns:
        - DataFrame: Preprocessed features
        """
        return self._preprocess_x_data(x_df, split_name="foo")



    # ------------------------
    #     Helper Methods
    # ------------------------

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
            for prefix, fill_value in self.preprocessing_config['fill_na'].items():
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


    def preprocess_y_data(self, y_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Preprocess target data including classification threshold transformation if needed.

        Also used in model_evaluation.py.

        Params:
        - y_df (DataFrame): Target variable DataFrame with single column
        - split_name (str): Name of split for error reporting and logging

        Returns:
        - DataFrame: Preprocessed target data (continuous or binary based on model_type)
        """
        if len(y_df.columns) != 1:
            raise ValueError(f"Target DataFrame should have exactly 1 column, "
                            f"found {len(y_df.columns)} in {split_name}")

        y_processed = y_df.copy()
        model_type = self.modeling_config['training']['model_type']

        # Apply classification threshold if needed
        if model_type == 'classification':

            # Convert continuous target to binary
            threshold = self.modeling_config['target']['classification']['threshold']
            original_values = y_processed.iloc[:, 0]
            binary_values = (original_values > threshold).astype(int)
            y_processed.iloc[:, 0] = binary_values

            # Calculate and log class distribution
            class_1_count = binary_values.sum()
            class_0_count = len(binary_values) - class_1_count
            class_1_pct = (class_1_count / len(binary_values)) * 100

            logger.info(f"Applied classification threshold {threshold} to {split_name}: "
                    f"{class_1_count:,} positive ({class_1_pct:.1f}%), "
                    f"{class_0_count:,} negative ({100-class_1_pct:.1f}%)")
            if class_1_pct < 5:
                logger.warning(f"Class imbalance in {split_name} below 5%: "
                        f"{class_1_pct:.1f}% positive class")

        # For regression, just ensure numeric type
        elif model_type == 'regression':
            if not pd.api.types.is_numeric_dtype(y_processed.iloc[:, 0]):
                raise ValueError(f"Target column must be numeric for regression")

        # Convert to float32 for consistency with X data
        y_processed = y_processed.astype('float32')

        return y_processed



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


    def _compile_training_metadata(self, training_data: dict) -> dict:
        """
        Compile metadata about training configuration and feature columns.

        Params:
        - training_data (dict): Full training data dictionary with all splits

        Returns:
        - dict: Complete metadata including config and column information
        """
        x_train_df = training_data['x_train']
        y_train_df = training_data['y_train']
        target_column_name = y_train_df.columns[0]

        metadata = {
            'sage_wallets_config': self.wallets_config,
            'feature_columns': x_train_df.columns.tolist(),
            'feature_count': len(x_train_df.columns),
            'target_variable': target_column_name,
            'preprocessing_timestamp': pd.Timestamp.now().isoformat()
        }

        return metadata


    def _save_preprocessed_df(self, df: pd.DataFrame, split_name: str) -> None:
        """
        Save a single preprocessed DataFrame to local CSV file in date-specific folder.
        Format matches exactly what gets uploaded to S3 for SageMaker.
        """
        if not hasattr(self, 'date_suffix') or not self.date_suffix:
            raise ValueError("date_suffix must be set before saving preprocessed data")

        # Create date-specific folder structure matching S3 upload pattern
        date_folder = self.output_base / self.date_suffix
        date_folder.mkdir(exist_ok=True)

        filename = f"{split_name}.csv"
        filepath = date_folder / filename

        df.to_csv(filepath, index=False, header=False)
        logger.info(f"Saved preprocessed {split_name} split to {filepath}")
