"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import tempfile
import os
from pathlib import Path
import pandas as pd
import boto3
from botocore.exceptions import ClientError


# Local modules
from sagemaker_wallets.wallet_modeler import WalletModeler

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletWorkflowOrchestrator:
    """
    Prepares data and orchestrates multiple instances of WalletModeler to build
     models for all provided dates.

    Params:
    """
    def __init__(self, sagewallets_config: dict):

        # Config
        self.sagewallets_config = sagewallets_config

        # Training data variables
        self.training_data = None
        self.data_folder = None



    # ------------------------
    #      Public Methods
    # ------------------------

    def load_training_data(
            self,
            date_suffixes: list
        ):
        """
        Load and combine training data across multiple prediction period dates.

        Files are loaded from sagewallets_config.training_data.local_load_folder.

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
        # Data location validation
        load_folder = self.sagewallets_config['training_data']['local_load_folder']
        self.data_folder = Path('../s3_uploads') / 'wallet_training_data_queue' / load_folder
        self._validate_data_folder()

        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        combined_data = {}

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


    def upload_training_data(self, overwrite_existing: bool = False):
        """
        Upload all training data splits to S3.

        Params:
        - overwrite_existing (bool): If True, overwrites existing S3 objects

        Returns:
        - dict: S3 URIs for each data split
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_training_data() first.")

        s3_client = boto3.client('s3')
        bucket_name = self.sagewallets_config['aws']['training_bucket']
        base_folder = 'training_data_processed'
        folder_prefix = f"{self.sagewallets_config['training_data']['upload_folder']}/"

        # Calculate total data size for confirmation
        total_size_bytes = sum(df.memory_usage(deep=True).sum() for df in self.training_data.values())
        total_size_gb = total_size_bytes / (1024**3)
        total_rows = sum(len(df) for df in self.training_data.values())
        total_files = len(self.training_data)

        # Confirmation prompt
        logger.info(f"Ready to upload {total_files} training data files ({total_size_gb:.2f}GB) "
                    f"with {total_rows:,} rows.")
        logger.info(f"Target: s3://{bucket_name}/{base_folder}/{folder_prefix}")
        confirmation = input("Proceed with upload? (y/N): ")

        if confirmation.lower() != 'y':
            logger.info("Upload cancelled")
            return {}

        s3_uris = {}

        for split_name, df in self.training_data.items():
            s3_key = f"{base_folder}/{folder_prefix}{split_name}.csv"
            s3_uri = f"s3://{bucket_name}/{s3_key}"

            # Check if file exists
            if not overwrite_existing:
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    logger.info(f"File {s3_key} already exists, skipping upload")
                    s3_uris[split_name] = s3_uri
                    continue
                except ClientError:
                    pass  # File doesn't exist, proceed with upload

            # Upload file
            logger.info(f"Uploading file {s3_key}")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, header=False, index=False)
                temp_file_path = temp_file.name

            s3_client.upload_file(temp_file_path, bucket_name, s3_key)
            os.unlink(temp_file_path)

            s3_uris[split_name] = s3_uri
            logger.info(f"Uploaded {split_name} to {s3_uri}")

        return s3_uris


    def run_training_pipeline(self):
        """
        Trains models for all configured scenarios.
        """
        modeler = WalletModeler(self.sagewallets_config, self.training_data)


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


