"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import tempfile
import os
from pathlib import Path
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError


# Local modules
from sagemaker_wallets.wallet_modeler import WalletModeler
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletWorkflowOrchestrator:
    """
    Prepares data and orchestrates multiple instances of WalletModeler to build
     models for all provided dates.

    Params:
    """
    def __init__(self, sage_wallets_config: dict):

        # Config
        self.sage_wallets_config = sage_wallets_config

        # Training data variables
        self.training_data = None
        self.data_folder = None
        self.date_suffixes = None



    # ------------------------
    #      Public Methods
    # ------------------------

    def load_training_data(
            self,
            date_suffixes: list
        ):
        """
        Load and combine training data across multiple prediction period dates.

        Files are loaded from sage_wallets_config.training_data.local_load_folder.

        Params:
        - date_suffixes (list): List of date suffixes (e.g., ["250301", "250401"])

        Data Split Usage Summary
        -----------------------
        X_train/y_train: Primary training data for model fitting
        X_eval/y_eval: Early stopping validation during XGBoost training (prevents overfitting)
        X_test/y_test: Hold-out test set for final model evaluation (traditional ML validation)
        X_validation/y_validation: Future time period data for realistic performance assessment

        Key Interactions:
        The Test set ML metrics (accuracy, R², etc.) are based on data from the same period
         as the Train set.
        The Validation set metrics are based on data from the future period just after the
         base y_train period ends. The Validation set represents actual future data the model
         would see in production, and Validation metrics measure model performance in a real
         world scenario.
        """
        # Data location validation with dataset suffix
        load_folder = self.sage_wallets_config['training_data']['local_load_folder']
        dataset = self.sage_wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            load_folder = f"{load_folder}_dev"

        self.data_folder = Path('../s3_uploads') / 'wallet_training_data_queue' / load_folder
        self._validate_data_folder()

        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        # Store date suffixes for upload method
        self.date_suffixes = date_suffixes

        combined_data = {}

        logger.milestone(f"<{dataset.upper()}> Loading training data for {len(date_suffixes)} "
                    f"periods: {date_suffixes}")
        for i, date_suffix in enumerate(date_suffixes):
            period_data = self._load_single_date_data(date_suffix)

            if i == 0:
                # Initialize with first period's data
                combined_data = period_data.copy()
            else:
                # Concatenate each DataFrame with matching key
                for key, df in period_data.items():
                    combined_data[key] = pd.concat([combined_data[key], df], ignore_index=True)

        self.training_data = combined_data

        # Success logging with data shape summary
        total_rows = sum(df.shape[0] for df in combined_data.values())
        data_splits = list(combined_data.keys())
        logger.info(f"Training data loaded successfully: {len(data_splits)} splits, "
                    f"{total_rows:,} total rows")

        # Log individual split sizes for debugging
        for split_name, df in combined_data.items():
            logger.debug(f"  {split_name}: {df.shape[0]:,} rows × {df.shape[1]} cols")


    def upload_training_data(self, overwrite_existing: bool = False):
        """
        Upload training data splits to S3, organized by date suffix folders.

        Params:
        - overwrite_existing (bool): If True, overwrites existing S3 objects
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_training_data() first.")

        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Ensure load_training_data() completed "
                             "successfully.")

        s3_client = boto3.client('s3')
        bucket_name = self.sage_wallets_config['aws']['training_bucket']
        base_folder = 'training-data-raw'

        upload_folder = self.sage_wallets_config['training_data']['upload_folder']
        dataset = self.sage_wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            upload_folder = f"{upload_folder}_dev"

        folder_prefix = f"{upload_folder}/"

        # Calculate total upload size for confirmation
        total_files = len(self.date_suffixes) * 8  # 8 files per date (x_train, y_train, etc.)

        logger.info(f"<{dataset.upper}> Ready to upload {total_files} training data files "
                    "across {len(self.date_suffixes)} date folders.")
        logger.info(f"Target: s3://{bucket_name}/{base_folder}/{folder_prefix}[DATE]/")
        confirmation = input("Proceed with upload? (y/N): ")

        if confirmation.lower() != 'y':
            logger.info("Upload cancelled")
            return {}

        for date_suffix in self.date_suffixes:
            # Load data for this specific date
            period_data = self._load_single_date_data(date_suffix)
            date_uris = {}

            for split_name, df in period_data.items():
                s3_key = f"{base_folder}/{folder_prefix}{date_suffix}/{split_name}.csv"
                s3_uri = f"s3://{bucket_name}/{s3_key}"

                # Check if file exists
                if not overwrite_existing:
                    try:
                        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        logger.info(f"File {s3_key} already exists, skipping upload")
                        date_uris[split_name] = s3_uri
                        continue
                    except ClientError:
                        pass  # File doesn't exist, proceed with upload

                # Validate CSV safety before upload
                self._validate_csv_safety(df, split_name)

                # Upload file
                logger.info(f"Uploading file {s3_key}")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    df.to_csv(temp_file.name, header=False, index=False)
                    temp_file_path = temp_file.name

                s3_client.upload_file(temp_file_path, bucket_name, s3_key)
                os.unlink(temp_file_path)

                date_uris[split_name] = s3_uri
                logger.info(f"Uploaded {split_name} to {s3_uri}")



    def retrieve_training_data_uris(self, date_suffixes: list):
        """
        Generate S3 URIs for training data without uploading.
        Uses same logic as upload_training_data() for consistency.

        Params:
        - date_suffixes (list): List of date suffixes (e.g., ["231107", "231201"])

        Returns:
        - dict: S3 URIs for each date suffix and data split
        """
        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        bucket_name = self.sage_wallets_config['aws']['training_bucket']
        base_folder = 'training-data-raw'

        upload_folder = self.sage_wallets_config['training_data']['upload_folder']
        dataset = self.sage_wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            upload_folder = f"{upload_folder}_dev"

        folder_prefix = f"{upload_folder}/"

        s3_uris = {}
        splits = ['x_train', 'y_train', 'x_test', 'y_test', 'x_eval', 'y_eval', 'x_val', 'y_val']

        for date_suffix in date_suffixes:
            date_uris = {}

            for split_name in splits:
                s3_key = f"{base_folder}/{folder_prefix}{date_suffix}/{split_name}.csv"
                s3_uri = f"s3://{bucket_name}/{s3_key}"
                date_uris[split_name] = s3_uri

            s3_uris[date_suffix] = date_uris

        return s3_uris



    def run_training_pipeline(self):
        """
        Trains models for all configured scenarios.
        """
        modeler = WalletModeler(self.sage_wallets_config, self.training_data)


    def run_scoring_pipeline(self):
        """
        Scores all models from configured scenarios.
        """
        pass







    # ------------------------
    #      Helper Methods
    # ------------------------

    def _load_single_date_data(self, date_suffix: str):
        """
        Load training data for a specific prediction period date.

        Params:
        - date_suffix (str): Date suffix for file selection (e.g., "250301")

        Returns:
        - dict: Contains X and y DataFrames for train/test/eval/val splits
        """
        data = {}

        # Define file patterns
        splits = ['train', 'test', 'eval', 'val']
        data_types = ['x', 'y']

        for data_type in data_types:
            for split in splits:
                # Build filename pattern
                pattern = f"{data_type}_{split}*{date_suffix}.parquet"
                matching_files = list(self.data_folder.glob(pattern))

                if not matching_files:
                    raise FileNotFoundError(
                        f"No file found matching pattern '{pattern}' in {self.data_folder}"
                    )

                if len(matching_files) > 1:
                    raise ValueError(
                        f"Multiple files found for pattern '{pattern}': {[f.name for f in matching_files]}"
                    )

                # Load the parquet file
                key = f"{data_type}_{split}"
                data[key] = pd.read_parquet(matching_files[0])

        # Validate X-y index consistency for each split
        for split in splits:
            x_key = f"x_{split}"
            y_key = f"y_{split}"

            if not np.array_equal(data[x_key].index.values, data[y_key].index.values):
                raise ValueError(
                    f"Index mismatch between {x_key} and {y_key} for date {date_suffix}. "
                    f"Shapes: {x_key}={data[x_key].shape}, {y_key}={data[y_key].shape}"
                )

        # Validate column consistency across all X DataFrames
        for split in ['test', 'eval', 'val']:
            x_key = f"x_{split}"
            try:
                u.validate_column_consistency(data['x_train'], data[x_key])
            except ValueError as e:
                raise ValueError(f"Column consistency failed between x_train and {x_key} "
                                 f"for date {date_suffix}: {str(e)}") from e

        return data


    def _validate_data_folder(self):
        """
        Validates that data folder exists and contains required parquet files.
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder does not exist: {self.data_folder}")

        if not self.data_folder.is_dir():
            raise NotADirectoryError(f"Data folder path is not a directory: {self.data_folder}")

        required_prefixes = [
            'x_test', 'x_train', 'x_eval', 'x_val',
            'y_test', 'y_train', 'y_eval', 'y_val'
        ]

        parquet_files = list(self.data_folder.glob('*.parquet'))

        for prefix in required_prefixes:
            matching_files = [f for f in parquet_files if f.name.startswith(prefix)]
            if not matching_files:
                raise FileNotFoundError(
                    f"No parquet file found starting with '{prefix}' in {self.data_folder}"
                )


    def _validate_csv_safety(self, df: pd.DataFrame, split_name: str):
        """
        Check for CSV-unsafe characters in DataFrame before upload.

        Params:
        - df (DataFrame): DataFrame to validate
        - split_name (str): Name of the data split for error reporting
        """
        string_cols = df.select_dtypes(include=['object']).columns

        for col in string_cols:
            if df[col].astype(str).str.contains('[,"\n\r]', na=False).any():
                problematic_values = df[col][df[col].astype(str).str.contains('[,"\n\r]', na=False)]
                raise ValueError(
                    f"CSV-unsafe characters found in {split_name}.{col}: {problematic_values.iloc[0]}"
                )
