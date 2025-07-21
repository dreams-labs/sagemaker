"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Local modules
from sage_wallet_modeling.wallet_preprocessor import SageWalletsPreprocessor
from sage_wallet_modeling.wallet_modeler import WalletModeler
import utils as u
import sage_utils.config_validation as ucv

@dataclass
class UploadContext:
    """
    Holds configuration and metadata for S3 uploads of preprocessed training data.
    """
    bucket_name: str
    base_folder: str
    folder_prefix: str
    total_size_mb: float
    total_rows: int
    dataset: str
    overwrite_existing: bool


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
        self.dataset = self.wallets_config['training_data'].get('dataset', 'prod')

        # Training data variables
        self.training_data = None
        self.data_folder = None
        self.date_suffixes = None



    # ------------------------
    #      Public Methods
    # ------------------------

    def load_all_training_data(
        self,
        date_suffixes: list
        ):
        """
        Load training data for multiple prediction period dates, maintaining separate
        datasets for each date suffix.

        Files are loaded from wallets_config.training_data.local_directory. Each date
        suffix represents a distinct modeling period with its own train/test/eval/val
        splits that should be processed independently.

        Params:
        - date_suffixes (list): List of date suffixes (e.g., ["250301", "250401"])

        Returns:
        - Sets self.training_data to nested dict structure:
            {
                "250301": {x_train, y_train, x_test, y_test, x_eval, y_eval, x_val, y_val},
                "250401": {x_train, y_train, x_test, y_test, x_eval, y_eval, x_val, y_val}
            }

        Data Split Usage Summary
        -----------------------
        X_train/y_train: Primary training data for model fitting
        X_eval/y_eval: Early stopping validation during XGBoost training (prevents overfitting)
        X_test/y_test: Hold-out test set for final model evaluation (traditional ML validation)
        X_validation/y_validation: Future time period data for realistic performance assessment

        Note: Each date suffix maintains independent data splits. Offset records have
        already been merged upstream, so no concatenation occurs at this stage.
        """
        # Data location validation with dataset suffix
        load_folder = self.wallets_config['training_data']['local_directory']

        if self.dataset == 'dev':
            load_folder = f"{load_folder}_dev"

        self.data_folder = Path('../s3_uploads') / 'wallet_training_data_queue' / load_folder
        self._validate_data_folder()

        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        # Store date suffixes for upload method
        self.date_suffixes = date_suffixes

        training_data_by_date = {}

        logger.milestone(f"<{self.dataset.upper()}> Loading training data for {len(date_suffixes)} "
                    f"periods: {date_suffixes}")
        for date_suffix in date_suffixes:
            period_data = self._load_single_date_data(date_suffix)
            training_data_by_date[date_suffix] = period_data

        self.training_data = training_data_by_date

        # Success logging with data shape summary
        total_rows = sum(
            df.shape[0]
            for date_data in training_data_by_date.values()
            for df in date_data.values()
        )
        offsets_per_df = len(self.training_data[date_suffixes[0]]
                             ['x_train'].index.get_level_values('epoch_start_date').unique())
        logger.info(f"Training data loaded successfully: {total_rows:,} total rows "
                    f"and {offsets_per_df} offsets for each date_suffix.")

        # Log individual date sizes for debugging
        for date_suffix, date_data in training_data_by_date.items():
            date_rows = sum(df.shape[0] for df in date_data.values())
            logger.debug(f"  {date_suffix}: {date_rows:,} rows across {len(date_data)} splits")


    def preprocess_all_training_data(self):
        """
        Preprocess training data for all loaded date suffixes independently.

        Each date suffix gets its own preprocessing run to maintain temporal
        boundaries and avoid data leakage between modeling periods.

        Returns:
        - dict: Preprocessed data keyed by date suffix
        {
            "250301": {train, test, eval, val, metadata},
            "250401": {train, test, eval, val, metadata}
        }

        Raises:
        - ValueError: If training_data not loaded or preprocessor unavailable
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_all_training_data() first.")

        preprocessed_by_date = {}

        logger.info(f"Preprocessing {len(self.training_data)} date periods...")

        for date_suffix, date_data in self.training_data.items():
            logger.debug(f"Preprocessing data for {date_suffix}...")

            # Initialize preprocessor for this date
            preprocessor = SageWalletsPreprocessor(self.wallets_config)

            # Preprocess this date's data
            preprocessed_data = preprocessor.preprocess_training_data(date_data, date_suffix)

            # Store results
            preprocessed_by_date[date_suffix] = preprocessed_data

            # Log preprocessing results
            total_rows = sum(
                df.shape[0] for df in preprocessed_data.values()
                if isinstance(df, pd.DataFrame)
            )
            logger.debug(f"  {date_suffix}: {total_rows:,} preprocessed rows")

        logger.info(f"Preprocessing complete for all {len(preprocessed_by_date)} dates.")


    def upload_training_data(self, overwrite_existing: bool = False):
        """
        Upload preprocessed training data splits to S3, organized by date suffix folders.
        Reads from saved CSV files to ensure perfect consistency with local artifacts.

        Params:
        - overwrite_existing (bool): If True, overwrites existing S3 objects
        """
        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Ensure load_all_training_data() completed "
                            "successfully.")

        # Load preprocessed data from saved CSV files
        preprocessed_data_by_date = self._load_preprocessed_training_data(self.date_suffixes)

        # Initialize UploadContext for this upload operation
        context = self._prepare_upload_context(preprocessed_data_by_date, overwrite_existing)

        # Confirm upload based on prepared context
        if not self._confirm_upload(context):
            return {}

        logger.info("Beginning approved upload...")

        s3_client = boto3.client('s3')
        upload_results = {}

        for date_suffix in self.date_suffixes:
            # Upload CSV splits for this date
            date_uris = self._upload_csv_files(
                date_suffix,
                preprocessed_data_by_date[date_suffix],
                context,
                s3_client
            )

            # Upload metadata for this date (metadata still comes from preprocessing step)
            metadata_uri = self._upload_metadata_for_date(
                date_suffix,
                {},  # Empty metadata since we're reading from files
                context,
                s3_client
            )
            date_uris['metadata'] = metadata_uri

            upload_results[date_suffix] = date_uris

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
                        if obj['Key'].split('/')[-1].startswith(f"{split_name}") and obj['Key'].endswith('.csv')
                    ]

                    if len(matching_objects) == 0:
                        raise FileNotFoundError(f"No CSV files found starting with '{split_name}' "
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

    def _prepare_upload_context(
            self,
            preprocessed_data_by_date: dict,
            overwrite_existing: bool
        ) -> UploadContext:
        """
        Prepare UploadContext with all S3 paths, size, and settings.

        Params:
        - preprocessed_data_by_date (dict): dict keyed by date_suffix containing DataFrames & metadata
        - overwrite_existing (bool): whether to overwrite existing S3 objects

        Returns:
        - UploadContext: configured upload context
        """
        # Get S3 upload paths
        bucket_name, base_folder, folder_prefix = self._get_s3_upload_paths()

        # Compute total upload size (MB) across all dates
        total_size_mb = (
            sum(
                df.memory_usage(deep=True).sum()
                for date_data in preprocessed_data_by_date.values()
                for df in date_data.values()
                if isinstance(df, pd.DataFrame)
            )
            / (1024 * 1024)
        )

        # Compute total rows across all DataFrames and date suffixes
        total_rows = sum(
            len(df)
            for date_data in preprocessed_data_by_date.values()
            for df in date_data.values()
            if isinstance(df, pd.DataFrame)
        )

        # Determine dataset tag
        dataset = self.wallets_config['training_data'].get('dataset', 'prod')

        return UploadContext(
            bucket_name=bucket_name,
            base_folder=base_folder,
            folder_prefix=folder_prefix,
            total_size_mb=total_size_mb,
            total_rows=total_rows,
            dataset=dataset,
            overwrite_existing=overwrite_existing
        )


    def _load_preprocessed_training_data(self, date_suffixes: list) -> dict:
        """
        Load preprocessed CSV files for upload, ensuring perfect consistency
        between saved files and uploaded data.

        Returns:
        - dict: Keyed by date_suffix, containing split DataFrames
        """
        # Get the preprocessed data directory from SageWalletsPreprocessor logic
        base_dir = (Path(f"{self.wallets_config['training_data']['local_s3_uploads_root']}")
                    / "wallet_training_data_preprocessed")
        local_dir = self.wallets_config["training_data"]["local_directory"]
        preprocessed_dir = base_dir / local_dir

        splits = ['train', 'test', 'eval', 'val']
        data_by_date = {}

        for date_suffix in date_suffixes:
            date_data = {}
            for split_name in splits:
                filename = f"{split_name}_preprocessed_{date_suffix}.csv"
                filepath = preprocessed_dir / filename

                if not filepath.exists():
                    raise FileNotFoundError(f"Preprocessed file not found: {filepath}")

                date_data[split_name] = pd.read_csv(filepath, header=None)

            data_by_date[date_suffix] = date_data

        return data_by_date


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
        base_folder = self.wallets_config['aws']['preprocessed_directory']

        upload_directory = self.wallets_config['training_data']['upload_directory']
        dataset = self.wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            upload_directory = f"{upload_directory}-dev"

        folder_prefix = f"{upload_directory}/"

        return bucket_name, base_folder, folder_prefix


    def _confirm_upload(self, context: UploadContext) -> bool:
        """
        Prompt user to confirm upload with summary logs.

        Params:
        - context (UploadContext): context containing paths and size

        Returns:
        - bool: True if user confirms, False otherwise
        """
        logger.milestone(
            f"<{context.dataset.upper()}> Ready to upload "
            f"{context.total_rows} total rows "
            f"({context.total_size_mb:.1f} MB) of preprocessed training data "
            f"across {len(self.date_suffixes)} date folders."
        )

        # If overwrite allowed, check for existing files to determine if an overwrite will actually happen
        if context.overwrite_existing:
            s3_client = boto3.client("s3")
            for suffix in self.date_suffixes:
                prefix = f"{context.base_folder}/{context.folder_prefix}{suffix}/"
                resp = s3_client.list_objects_v2(
                    Bucket=context.bucket_name, Prefix=prefix, MaxKeys=1
                )
                if resp.get("KeyCount", 0) > 0:
                    logger.milestone("This upload will overwrite existing files.")
                    break

        # Log upload information and request approval
        logger.info(
            f"Target: s3://{context.bucket_name}/"
            f"{context.base_folder}/{context.folder_prefix}[DATE]/"
        )
        confirmation = input("Proceed with upload? (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Upload cancelled")
            return False
        return True

    def _upload_csv_files(
        self,
        date_suffix: str,
        preprocessed_data: dict,
        context: UploadContext,
        s3_client
    ) -> dict[str, str]:
        """
        Upload all DataFrame splits (excluding metadata) for one date to S3.
        Returns a mapping of split_name to S3 URI.
        """
        date_uris: dict[str, str] = {}
        for split_name, df in preprocessed_data.items():
            if split_name == 'metadata':
                continue

            filename = f"{split_name}.csv"
            s3_key = f"{context.base_folder}/{context.folder_prefix}{date_suffix}/{filename}"
            s3_uri = f"s3://{context.bucket_name}/{s3_key}"

            if not context.overwrite_existing:
                try:
                    s3_client.head_object(Bucket=context.bucket_name, Key=s3_key)
                    logger.info(f"File {s3_key} exists, skipping upload")
                    date_uris[split_name] = s3_uri
                    continue
                except ClientError:
                    pass  # Doesn't exist yet

            self._validate_csv_safety(df, split_name)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                df.to_csv(tmp.name, index=False, header=False)
                temp_path = tmp.name

            s3_client.upload_file(temp_path, context.bucket_name, s3_key)
            os.unlink(temp_path)
            logger.info(f"Uploaded {split_name} to {s3_uri}")
            date_uris[split_name] = s3_uri

        return date_uris

    def _upload_metadata_for_date(
        self,
        date_suffix: str,
        metadata: dict,
        context: UploadContext,
        s3_client
    ) -> str:
        """
        Upload metadata JSON for one date to S3.
        Returns the metadata file's S3 URI.
        """
        filename = "metadata.json"
        s3_key = f"{context.base_folder}/{context.folder_prefix}{date_suffix}/{filename}"
        s3_uri = f"s3://{context.bucket_name}/{s3_key}"

        if not context.overwrite_existing:
            try:
                s3_client.head_object(Bucket=context.bucket_name, Key=s3_key)
                logger.info(f"Metadata file {s3_key} exists, skipping upload")
                return s3_uri
            except ClientError:
                pass  # Doesn't exist yet

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metadata, tmp, indent=2)
            temp_path = tmp.name

        s3_client.upload_file(temp_path, context.bucket_name, s3_key)
        os.unlink(temp_path)
        logger.info(f"Uploaded metadata to {s3_uri}")
        return s3_uri
