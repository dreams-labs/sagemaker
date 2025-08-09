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

        # Set up local output directory for this run
        base_dir = (Path(f"{self.wallets_config['training_data']['local_s3_root']}")
                    / "s3_uploads"
                    / "wallet_training_data_preprocessed")
        if not base_dir.exists():
            raise FileNotFoundError(f"Expected preprocessed data base directory not found: {base_dir}")
        self.output_base = base_dir / self.wallets_config["training_data"]["local_directory"]
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

            # Preprocess X and y data
            x_preprocessed = self._preprocess_x_data(training_data[x_split], x_split)
            y_preprocessed = training_data[y_split]
            processed_data[f"{split_name}_y"] = y_preprocessed

            # Append additional column
            x_preprocessed = self._retain_offset_date(x_preprocessed)

            # Store preprocessed data in our results dict
            if split_name in ['train', 'eval']:
                # For custom transforms, keep X and y separate (X only for train/eval)
                processed_data[split_name] = x_preprocessed

            else:
                # Test and Val sets are only used for scoring, and shouldn't have targets
                processed_data[split_name] = x_preprocessed

            # Save preprocessed X data to local file
            self._save_preprocessed_df(processed_data[split_name], split_name)

            # Save preprocessed y data to local file
            self._save_preprocessed_df(processed_data[f"{split_name}_y"], f"{split_name}_y")

            logger.info(f"Preprocessed {split_name}: {processed_data[split_name].shape[0]:,} rows "
                        f"× {processed_data[split_name].shape[1]} cols.")

        # Save metadata alongside CSV files
        processed_data['metadata'] = self._compile_training_metadata(
            processed_data['train'].copy(),
            processed_data['train_y'].copy())
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
        # Ensure numeric data types
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Non-numeric columns found in {split_name}: "
                             f"{list(non_numeric_cols)}")

        # Convert all to float64 for consistency and null handling
        df = df.astype('float64')

        # Handle missing values with intelligent fill strategy
        df = self._handle_missing_values(df, split_name)

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


    def _retain_offset_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract epoch_start_date from index and prepend as first column for temporal filtering.
        Converts dates to days since configured reference date for numeric compatibility.

        Params:
        - df (DataFrame): Preprocessed features with MultiIndex containing epoch_start_date

        Returns:
        - DataFrame: Features with offset_date as first column (days since date_0)
        """
        # Get reference date from config (format: YYMMDD string)
        date_0_str = str(self.wallets_config['training_data']['date_0'])

        # Parse YYMMDD format to datetime
        reference_date = pd.to_datetime(date_0_str, format='%y%m%d')

        # Extract epoch_start_date from the MultiIndex
        offset_dates = df.index.get_level_values('epoch_start_date')

        # Convert to days since reference date (as float32 for XGBoost)
        # This gives us integers like 30, 60, 90 for monthly offsets
        offset_days = (offset_dates - reference_date).days.astype('float32')

        # Reset index first to work with clean DataFrame
        df = df.reset_index(drop=True)

        # Insert offset_date as first column
        df.insert(0, 'offset_date', offset_days)

        # Log the range of offset dates for debugging
        logger.debug(f"Offset dates range: {offset_days.min():.0f} to {offset_days.max():.0f} "
                    f"days from {reference_date.strftime('%Y-%m-%d')}")

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


    def _compile_training_metadata(
            self,
            processed_x: pd.DataFrame,
            processed_y: pd.DataFrame
        ) -> dict:
        """
        Compile metadata about training configuration and feature columns.

        Params:
        - processed_x (DataFrame): Preprocessed features DataFrame
        - processed_y (DataFrame): Preprocessed target DataFrame

        Returns:
        - dict: Complete metadata including config and column information
        """
        target_column_name = processed_y.columns[0]

        metadata = {
            'sage_wallets_config': self.wallets_config,
            'feature_columns': processed_x.columns.tolist(),
            'feature_count': len(processed_x.columns),
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

         # Guard: do not write files with NaNs
        if df.isnull().values.any():
            cols_with_na = df.columns[df.isnull().any()].tolist()
            logger.error(
                "NaN values detected in preprocessed '%s' split; columns with NaNs: %s",
                split_name, cols_with_na
            )
            raise ValueError(
                f"Cannot save preprocessed '{split_name}' split: NaN values present in "
                f"columns {cols_with_na}"
            )

        # Create date-specific folder structure matching S3 upload pattern
        date_folder = self.output_base / self.date_suffix
        date_folder.mkdir(exist_ok=True)

        filename = f"{split_name}.csv"
        filepath = date_folder / filename

        df.to_csv(filepath, index=False, header=False)
        logger.info(f"Saved preprocessed {split_name} split to {filepath}")
