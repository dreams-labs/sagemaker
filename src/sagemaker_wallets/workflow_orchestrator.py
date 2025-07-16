"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import tempfile
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError


# Local modules
from sagemaker_wallets.wallet_modeler import WalletModeler
import utils as u
import sage_utils.config_validation as ucv

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletWorkflowOrchestrator:
    """
    Prepares data and orchestrates multiple instances of WalletModeler to build
     models for all provided dates.

    Params:
    - wallets_config (dict): abbreviated name for sage_wallets_config.yaml
    """
    def __init__(self, wallets_config: dict):

        # Config
        ucv.validate_sage_wallets_config(wallets_config)
        self.wallets_config = wallets_config

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

        Files are loaded from wallets_config.training_data.local_load_folder.

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
        load_folder = self.wallets_config['training_data']['local_load_folder']
        dataset = self.wallets_config['training_data'].get('dataset', 'prod')

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


    def upload_training_data(self, preprocessed_data: dict, overwrite_existing: bool = False):
        """
        Upload preprocessed training data splits to S3, organized by date suffix folders.
        Filenames include sanitized target variable names for metadata preservation.

        Params:
        - preprocessed_data (dict): Preprocessed data from SageWalletsPreprocessor
        - overwrite_existing (bool): If True, overwrites existing S3 objects
        """
        if not preprocessed_data:
            raise ValueError("No preprocessed data provided.")

        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Ensure load_training_data() completed "
                            "successfully.")

        # Get sanitized target variable name from y_train
        target_name = self.training_data['y_train'].columns[0]
        sanitized_target = target_name.replace('|', '_').replace('/', '_')

        # Identify all S3 path names
        bucket_name, base_folder, folder_prefix = self._get_s3_upload_paths()

        # Calculate total upload size for confirmation
        total_size_mb = sum(
            df.memory_usage(deep=True).sum() for df in preprocessed_data.values()
            if isinstance(df, pd.DataFrame)
        ) / (1024 * 1024) * len(self.date_suffixes)

        dataset = self.wallets_config['training_data'].get('dataset', 'prod')

        logger.milestone(f"<{dataset.upper()}> Ready to upload {total_size_mb:.1f} MB of preprocessed training data "
                    f"across {len(self.date_suffixes)} date folders.")
        logger.info(f"Target variable: {sanitized_target}")
        logger.info(f"Target: s3://{bucket_name}/{base_folder}/{folder_prefix}[DATE]/")
        confirmation = input("Proceed with upload? (y/N): ")

        if confirmation.lower() != 'y':
            logger.info("Upload cancelled")
            return {}

        s3_client = boto3.client('s3')
        upload_results = {}

        for date_suffix in self.date_suffixes:
            date_uris = {}

            for split_name, df in preprocessed_data.items():
                # Skip metadata - only process DataFrame splits
                if split_name == 'metadata':
                    continue

                # Enhanced filename with target variable
                filename = f"{split_name}_{sanitized_target}.csv"
                s3_key = f"{base_folder}/{folder_prefix}{date_suffix}/{filename}"
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
                logger.info(f"Uploading {split_name}_{sanitized_target} for {date_suffix}: {df.shape[0]:,} rows")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    df.to_csv(temp_file.name, header=False, index=False)
                    temp_file_path = temp_file.name

                s3_client.upload_file(temp_file_path, bucket_name, s3_key)
                os.unlink(temp_file_path)

                date_uris[split_name] = s3_uri
                logger.info(f"Uploaded {split_name} to {s3_uri}")

            upload_results[date_suffix] = date_uris

            # Upload metadata JSON for this date suffix
            metadata_filename = f"metadata_{sanitized_target}.json"
            metadata_s3_key = f"{base_folder}/{folder_prefix}{date_suffix}/{metadata_filename}"
            metadata_s3_uri = f"s3://{bucket_name}/{metadata_s3_key}"

            # Check if metadata file exists
            if not overwrite_existing:
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=metadata_s3_key)
                    logger.info(f"Metadata file {metadata_s3_key} already exists, skipping upload")
                except ClientError:
                    # File doesn't exist, proceed with upload
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        json.dump(preprocessed_data['metadata'], temp_file, indent=2)
                        temp_metadata_path = temp_file.name

                    s3_client.upload_file(temp_metadata_path, bucket_name, metadata_s3_key)
                    os.unlink(temp_metadata_path)
                    logger.info(f"Uploaded metadata to {metadata_s3_uri}")
            else:
                # Upload metadata (overwrite mode)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(preprocessed_data['metadata'], temp_file, indent=2)
                    temp_metadata_path = temp_file.name

                s3_client.upload_file(temp_metadata_path, bucket_name, metadata_s3_key)
                os.unlink(temp_metadata_path)
                logger.info(f"Uploaded metadata to {metadata_s3_uri}")

        return upload_results


    def retrieve_training_data_uris(self, date_suffixes: list):
        """
        Generate S3 URIs for training data by finding files that match split patterns.
        Validates files exist and handles target-variable-enhanced filenames.

        Params:
        - date_suffixes (list): List of date suffixes (e.g., ["231107", "231201"])

        Returns:
        - dict: S3 URIs for each date suffix and data split

        Raises:
        - FileNotFoundError: If any expected S3 objects don't exist
        - ValueError: If multiple files match the same split pattern
        """
        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        # Get S3 file locations
        bucket_name, base_folder, folder_prefix = self._get_s3_upload_paths()

        s3_client = boto3.client('s3')
        s3_uris = {}
        splits = ['train', 'test', 'eval', 'val']  # Preprocessed files combine x and y

        for date_suffix in date_suffixes:
            date_uris = {}

            for split_name in splits:
                # List objects with the split prefix
                prefix = f"{base_folder}/{folder_prefix}{date_suffix}/{split_name}"

                try:
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=prefix
                    )

                    if 'Contents' not in response:
                        raise FileNotFoundError("No S3 objects found matching prefix: "
                                                f"s3://{bucket_name}/{prefix}")

                    # Filter for CSV files that start with the exact split name
                    matching_objects = [
                        obj for obj in response['Contents']
                        if obj['Key'].split('/')[-1].startswith(f"{split_name}_") and obj['Key'].endswith('.csv')
                    ]

                    if len(matching_objects) == 0:
                        raise FileNotFoundError(f"No CSV files found starting with '{split_name}_' "
                                                f"at: s3://{bucket_name}/{prefix}")

                    if len(matching_objects) > 1:
                        matching_files = [obj['Key'].split('/')[-1] for obj in matching_objects]
                        raise ValueError(f"Multiple files found for split '{split_name}' in "
                                         f"{date_suffix}: {matching_files}")

                    # Use the actual filename found
                    s3_key = matching_objects[0]['Key']
                    s3_uri = f"s3://{bucket_name}/{s3_key}"
                    date_uris[split_name] = s3_uri

                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchBucket':
                        raise FileNotFoundError(f"S3 bucket does not exist: {bucket_name}") from e
                    else:
                        raise

            s3_uris[date_suffix] = date_uris

        return s3_uris

    def run_training_pipeline(self):
        """
        Trains models for all configured scenarios.
        """
        modeler = WalletModeler(self.wallets_config, self.training_data)


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


    def _get_s3_upload_paths(self) -> tuple[str, str, str]:
        """
        Get S3 bucket, base folder, and upload folder prefix for training data.

        Returns:
        - tuple: (bucket_name, base_folder, folder_prefix)
        """
        bucket_name = self.wallets_config['aws']['training_bucket']
        base_folder = self.wallets_config['aws']['preprocessed_folder']

        upload_folder = self.wallets_config['training_data']['upload_folder']
        dataset = self.wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            upload_folder = f"{upload_folder}-dev"

        folder_prefix = f"{upload_folder}/"

        return bucket_name, base_folder, folder_prefix
