"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import tempfile
import copy
from typing import List
import os
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures
import json
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Local modules
from sage_wallet_modeling.wallet_preprocessor import SageWalletsPreprocessor
from sage_wallet_modeling.wallet_modeler import WalletModeler
# For cross-date CV training
import sage_wallet_modeling.wallet_script_modeler as sm
import sage_wallet_insights.model_evaluation as sime
import script_modeling.custom_transforms as ct
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
    - modeling_config (dict): abbreviated name for sage_wallets_modeling_config.yaml
    """
    def __init__(self, wallets_config: dict, modeling_config: dict):

        # Configs
        ucv.validate_sage_wallets_config(wallets_config)
        ucv.validate_sage_wallets_modeling_config(modeling_config)
        self.wallets_config = wallets_config
        self.modeling_config = modeling_config
        self.dataset = self.wallets_config['training_data'].get('dataset', 'dev')


        # Training data variables
        self.training_data = None
        self.data_folder = None
        self.date_suffixes = None



    # ------------------------
    #      Public Methods
    # ------------------------

    def load_all_training_data(
        self
        ):
        """
        Load training data for multiple prediction period dates, maintaining separate
        datasets for each date suffix.

        Files are loaded from wallets_config.training_data.local_directory. Each date
        suffix represents a distinct modeling period with its own train/test/eval/val
        splits that should be processed independently.

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
        # Auto-detect date_suffixes from config
        split_requirements = calculate_comprehensive_offsets_by_split(self.wallets_config)
        train_offsets = split_requirements['all_train_offsets']
        eval_offsets = split_requirements['all_eval_offsets']
        test_offsets = split_requirements['all_test_offsets']
        val_offsets = split_requirements['all_val_offsets']

        # Combine all offsets and remove duplicates while preserving order
        all_offsets = train_offsets + eval_offsets + test_offsets + val_offsets
        date_suffixes = list(dict.fromkeys(all_offsets))  # Removes duplicates, preserves order

        logger.info(f"Auto-detected date_suffixes from config: {date_suffixes}")

        # Data location validation with dataset suffix
        load_folder = self.wallets_config['training_data']['training_data_directory']

        if self.dataset == 'dev':
            load_folder = f"{load_folder}_dev"

        self.data_folder = Path('../s3_uploads') / 'wallet_training_data_queue' / load_folder
        self._validate_data_folder()

        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty. Either provide explicit list or "
                            "configure train_offsets/eval_offsets/test_offsets/val_offsets in config.")

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

        If concatenate_offsets is enabled, filters each date_suffix to prevent
        temporal overlap during downstream concatenation.
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_all_training_data() first.")

        preprocessed_by_date = {}
        concatenate_mode = self.wallets_config['training_data'].get('concatenate_offsets', False)

        logger.info(f"Preprocessing {len(self.training_data)} date periods...")

        for date_suffix, date_data in self.training_data.items():
            logger.debug(f"Preprocessing data for {date_suffix}...")

            # Apply temporal filtering if concatenation mode enabled
            if concatenate_mode:
                date_data = self._filter_temporal_overlap(date_data, date_suffix)

            # Initialize preprocessor for this date
            preprocessor = SageWalletsPreprocessor(self.wallets_config, self.modeling_config)

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
        return preprocessed_by_date


    @u.timing_decorator
    def concatenate_all_preprocessed_data(self, repreprocess_offsets: bool = True) -> None:
        """
        Concatenate preprocessed CSVs across configured offsets for each split and
        save combined CSVs to the concatenated output directory.

        Applies fresh preprocessing with temporal filtering to prevent data overlap
        when concatenate_offsets is enabled.

        Exports separate y-files for test and val splits using RAW (non-preprocessed)
        y values for evaluation purposes.
        :param bool repreprocess_offsets: if False, skip re-running preprocessing and load existing CSVs.
        """
        # Preprocess all data from scratch to ensure temporal filtering is applied
        if repreprocess_offsets:
            data_by_date = self.preprocess_all_training_data()
        else:
            # Skip reprocessing and load saved preprocessed CSVs
            data_by_date = self._load_preprocessed_training_data(self.date_suffixes)

        logger.info("Beginning concatenation of preprocessed data...")
        # Determine offsets for each split from config
        split_requirements = calculate_comprehensive_offsets_by_split(self.wallets_config)

        offsets_map = {
            'train': split_requirements['all_train_offsets'],
            'eval':  split_requirements['all_eval_offsets'],
            'test':  split_requirements['all_test_offsets'],
            'val':   split_requirements['all_val_offsets']
        }

        # Build concatenation output directory alongside the preprocessed tree
        base_dir = Path(self.wallets_config['training_data']['local_s3_root']) \
                / "s3_uploads" \
                / "wallet_training_data_concatenated"
        local_dir = self.wallets_config["training_data"]["local_directory"]
        if self.dataset == 'dev':
            local_dir = f"{local_dir}_dev"
        concat_base = base_dir / local_dir
        concat_base.mkdir(parents=True, exist_ok=True)

        # Concatenate main splits
        for split, offsets in offsets_map.items():
            if not offsets:
                logger.warning(f"No offsets configured for split '{split}'")
                continue

            dfs = []
            for offset in offsets:
                if offset not in data_by_date:
                    raise KeyError(f"No data found for offset {offset}")
                if split not in data_by_date[offset]:
                    raise KeyError(f"No '{split}' split found for offset {offset}")
                df = data_by_date[offset][split]
                dfs.append(df)

            if not dfs:
                logger.warning(f"No data found for split '{split}' across offsets {offsets}")
                continue

            # Validate no NaNs before saving
            concatenated = pd.concat(dfs, ignore_index=True)
            if concatenated.isnull().any().any():
                raise ValueError(f"NaN values detected in {split} split before saving to CSV")

            out_file = concat_base / f"{split}.csv"
            concatenated.to_csv(out_file, index=False, header=False)
            logger.info(f"Saved concatenated {split}.csv with {len(concatenated)} rows to {out_file}")

        # Add this after the main concatenation loop, before the y-file exports
        logger.info("Copying metadata for concatenated dataset...")

        # Load metadata from first available date (they should all be identical)
        first_date_suffix = list(data_by_date.keys())[0]
        source_metadata_path = (
            Path(self.wallets_config['training_data']['local_s3_root'])
            / "s3_uploads"
            / "wallet_training_data_preprocessed"
            / local_dir
            / first_date_suffix
            / "metadata.json"
        )

        if not source_metadata_path.exists():
            logger.warning(f"No metadata found at {source_metadata_path}, skipping metadata copy")
        else:
            # Copy metadata to concatenated directory
            concat_metadata_path = concat_base / "metadata.json"

            with open(source_metadata_path, 'r', encoding='utf-8') as src:
                metadata = json.load(src)

            # Update metadata to reflect concatenated nature
            metadata['concatenated_from_dates'] = list(data_by_date.keys())
            metadata['concatenation_timestamp'] = pd.Timestamp.now().isoformat()

            with open(concat_metadata_path, 'w', encoding='utf-8') as dst:
                json.dump(metadata, dst, indent=2)

            logger.info(f"Saved concatenated metadata to {concat_metadata_path}")


        # Load raw training data for non-preprocessed y-values
        raw_data_by_date = {}
        for date_suffix in self.date_suffixes:
            raw_data_by_date[date_suffix] = self._load_single_date_data(date_suffix)

        # Always export full y for all splits with header
        y_splits_to_export = ['train', 'eval', 'test', 'val']
        include_header = True
        for split in y_splits_to_export:
            split_offsets = offsets_map.get(split, [])
            if not split_offsets:
                logger.warning(f"No offsets configured for y-file export of '{split}'")
                continue

            y_dfs = []
            y_split_key = f"y_{split}"

            for offset in split_offsets:
                if offset not in raw_data_by_date:
                    raise KeyError(f"No raw data found for offset {offset}")
                if y_split_key not in raw_data_by_date[offset]:
                    raise KeyError(f"No '{y_split_key}' split found for offset {offset}")

                # Use RAW y-data instead of preprocessed
                raw_y_df = raw_data_by_date[offset][y_split_key]
                y_dfs.append(raw_y_df)

            if not y_dfs:
                logger.warning(f"No raw y-data found for split '{split}' across offsets {split_offsets}")
                continue

            concatenated_y = pd.concat(y_dfs, ignore_index=True)
            y_out_file = concat_base / f"{split}_y.csv"
            concatenated_y.to_csv(y_out_file, index=False, header=include_header)
            logger.info(f"Saved concatenated {split}_y.csv with {len(concatenated_y)} "
                        f"rows to {y_out_file}")


    @u.timing_decorator
    def upload_concatenated_training_data(
            self,
            overwrite_existing: bool = False,
            splits: List = None
        ) -> dict[str, str]:
        """
        Upload concatenated training data splits to S3, organized under a single folder.
        Params:
        - overwrite_existing (bool): If True, overwrites existing S3 objects.
        - splits (list): which splits to upload
        Returns:
        - dict: Mapping of split_name to S3 URI for uploaded concatenated data.
        """
        # Assign to unique list
        if splits is None:
            splits = ['train', 'eval', 'test', 'val']

        # Always include y splits for upload
        y_splits = [f"{split}_y" for split in splits]
        splits = splits + y_splits

        # Determine S3 target paths
        bucket = self.wallets_config['aws']['training_bucket']
        # Use configured concatenated directory
        base_folder = self.wallets_config['aws']['concatenated_directory']
        upload_directory = self.wallets_config['training_data']['upload_directory']
        if self.dataset == 'dev':
            upload_directory = f"{upload_directory}-dev"
        folder_prefix = f"{upload_directory}/"

        s3_client = boto3.client('s3')
        upload_results = {}

        # Local concatenated directory
        concat_root = Path(self.wallets_config['training_data']['local_s3_root']) \
                      / "s3_uploads" \
                      / "wallet_training_data_concatenated"
        local_dir = self.wallets_config["training_data"]["local_directory"]
        if self.dataset == 'dev':
            local_dir = f"{local_dir}_dev"
        concat_dir = concat_root / local_dir

        logger.info(f"Beginning upload of concatenated training data for splits {splits}...")
        # Parallel upload of concatenated splits
        n_threads = self.wallets_config['n_threads']['upload_all_training_data']
        logger.info(f"Uploading concatenated splits in parallel with {n_threads} threads...")
        def _upload_split(split: str):
            s3_key = f"{base_folder}/{folder_prefix}{split}.csv"
            s3_uri = f"s3://{bucket}/{s3_key}"

            # Check if file exists
            try:
                s3_client.head_object(Bucket=bucket, Key=s3_key)

                # Escape if we're not overwriting
                if not overwrite_existing:
                    logger.info(f"File exists, skipping upload of concatenated split '{split}': {s3_key}")
                    upload_results[split] = s3_uri
                    return
                else:
                    logger.info(f"Overwriting existing file '{s3_uri}'...")
            # Error emitted if the file can't be found
            except ClientError:
                logger.info(f"Didn't find S3 file '{s3_uri}', proceeding with upload...")

            # Upload local file
            local_file = concat_dir / f"{split}.csv"
            if not local_file.exists():
                raise FileNotFoundError(f"Concatenated file not found: {local_file}")
            s3_client.upload_file(str(local_file), bucket, s3_key)
            logger.info(f"Uploaded concatenated split '{split}' to {s3_uri}")
            upload_results[split] = s3_uri

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            executor.map(_upload_split, splits)

        # Add this after the main CSV upload loop, before the return statement
        logger.info("Uploading concatenated metadata...")

        # Upload metadata.json
        metadata_file = concat_dir / "metadata.json"
        if metadata_file.exists():
            s3_key = f"{base_folder}/{folder_prefix}metadata.json"
            s3_uri = f"s3://{bucket}/{s3_key}"

            # Check if exists (following same pattern as splits)
            try:
                s3_client.head_object(Bucket=bucket, Key=s3_key)
                if not overwrite_existing:
                    logger.info(f"Metadata exists, skipping upload: {s3_key}")
                else:
                    logger.info(f"Overwriting existing metadata: {s3_uri}")
                    s3_client.upload_file(str(metadata_file), bucket, s3_key)
            except ClientError:
                logger.info(f"Uploading metadata to {s3_uri}")
                s3_client.upload_file(str(metadata_file), bucket, s3_key)

            upload_results['metadata'] = s3_uri
        else:
            logger.warning("No metadata.json found in concatenated directory")

        # Validate custom filters against metadata
        logger.info("Validating custom filter configuration...")
        self._validate_custom_filters_config(metadata_file)

        return upload_results


    def upload_all_training_data(self, overwrite_existing: bool = False):
        """
        Upload preprocessed training data splits to S3, organized by date suffix folders.
        Reads from saved CSV files to ensure perfect consistency with local artifacts.
        Uses multithreading to upload date folders in parallel.

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

        upload_results = {}
        n_threads = self.wallets_config['n_threads']['upload_all_training_data']

        logger.info(f"Uploading data for {len(self.date_suffixes)} date periods with {n_threads} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_date = {
                executor.submit(
                    self._upload_single_date,
                    date_suffix,
                    preprocessed_data_by_date[date_suffix],
                    context
                ): date_suffix
                for date_suffix in self.date_suffixes
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date_suffix = future_to_date[future]
                result = future.result()
                upload_results[date_suffix] = result

        logger.info(f"All {len(upload_results)} date uploads completed successfully.")
        return upload_results


    def retrieve_training_data_uris(self, date_suffixes: list):
        """
        Generate S3 URIs for training data by finding files that match split patterns.
        Handles both preprocessed (per-date folders) and concatenated (flat) structures.

        Params:
        - date_suffixes (list): List of date suffixes (e.g., ["231107", "231201"])
                            For concatenated data, use ["concat"]

        Returns:
        - dict: S3 URIs for each date suffix and data split
        """
        if not date_suffixes:
            raise ValueError("date_suffixes cannot be empty")

        # Check if we're using concatenated directory structure
        use_concatenated = self.wallets_config['training_data'].get('concatenate_offsets', False)

        # Get S3 file locations
        bucket_name, base_folder, folder_prefix = self._get_s3_upload_paths()

        s3_client = boto3.client('s3')
        s3_uris = {}
        splits = ['train', 'test', 'eval', 'val']

        for date_suffix in date_suffixes:
            date_uris = {}

            for split_name in splits:
                if use_concatenated:
                    # Concatenated structure: files directly under upload directory
                    # Path: s3://bucket/concatenated/{upload_dir}/train.csv
                    prefix = f"{base_folder}/{folder_prefix}{split_name}"
                    expected_filename = f"{split_name}.csv"
                else:
                    # Preprocessed structure: files in date-specific subdirectories
                    # Path: s3://bucket/preprocessed/{upload_dir}/{date_suffix}/train.csv
                    prefix = f"{base_folder}/{folder_prefix}{date_suffix}/{split_name}"
                    expected_filename = f"{split_name}"  # Original logic for preprocessed

                try:
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=prefix
                    )

                    if 'Contents' not in response:
                        logger.warning("No S3 objects found matching prefix: "
                                       f"s3://{bucket_name}/{prefix}")
                        continue

                    if use_concatenated:
                        # For concatenated, look for exact filename match
                        matching_objects = [
                            obj for obj in response['Contents']
                            if obj['Key'].split('/')[-1] == expected_filename
                        ]
                    else:
                        # Original logic for preprocessed files
                        matching_objects = [
                            obj for obj in response['Contents']
                            if obj['Key'].split('/')[-1].startswith(f"{split_name}") and obj['Key'].endswith('.csv')
                        ]

                    if len(matching_objects) == 0:
                        structure_type = "concatenated" if use_concatenated else "preprocessed"
                        raise FileNotFoundError(f"No {structure_type} CSV files found for '{split_name}' "
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


    def train_all_epoch_shift_models(
        self,
        concat_uris: dict[str, str]
    ) -> dict[int, dict]:
        """
        Train models for all epoch shifts using concatenated data with temporal filtering.

        Each model uses the same master concatenated dataset but applies different
        epoch_shift filtering in the container to train on different temporal windows.

        Params:
        - epoch_shifts (list[int]): List of shift values (e.g., [0, 30, 60, 90, 120])
        - concat_uris (dict): S3 URIs for concatenated data splits
            {'train': s3://..., 'eval': s3://..., 'train_y': s3://..., 'eval_y': s3://...}

        Returns:
        - dict: Training results keyed by epoch_shift {shift: training_result}
        """
        epoch_shifts = self.wallets_config['training_data']['epoch_shifts']
        if not epoch_shifts:
            raise ValueError("epoch_shifts cannot be empty")

        if not concat_uris:
            raise ValueError("concat_uris cannot be empty")

        training_results = {}
        n_threads = self.wallets_config['n_threads']['train_all_models']

        logger.milestone(f"Training models for {len(epoch_shifts)} epoch shifts: {epoch_shifts}")
        logger.info(f"Using {n_threads} concurrent training jobs")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_shift = {
                executor.submit(
                    self._train_single_epoch_shift,
                    shift,
                    concat_uris
                ): shift
                for shift in epoch_shifts
            }

            for future in concurrent.futures.as_completed(future_to_shift):
                shift = future_to_shift[future]
                try:
                    result = future.result()
                    training_results[shift] = result
                    logger.milestone(f"Successfully completed training for epoch_shift={shift}")
                except Exception as e:
                    logger.error(f"Training failed for epoch_shift={shift}: {e}")
                    training_results[shift] = {'error': str(e)}

        successful_models = len([r for r in training_results.values() if 'error' not in r])
        logger.milestone(f"Epoch shift training complete: {successful_models}/{len(epoch_shifts)} models successful")

        return training_results


    def _train_single_epoch_shift(
        self,
        epoch_shift: int,
        concat_uris: dict[str, str]
    ) -> dict:
        """
        Train a model for a specific epoch shift using concatenated data.

        Creates a modified modeling config with the epoch_shift hyperparameter,
        then launches training via WalletModeler with the concatenated dataset.

        Params:
        - epoch_shift (int): Days to shift epoch offsets (e.g., 0, 30, 60, 90, 120)
        - concat_uris (dict): S3 URIs for concatenated data splits

        Returns:
        - dict: Training results for this epoch shift
        """
        # Create modified modeling config with epoch_shift hyperparameter
        modified_modeling_config = copy.deepcopy(self.modeling_config)
        modified_modeling_config['training']['hyperparameters']['epoch_shift'] = epoch_shift

        # Prepare s3_uris dict keyed by synthetic suffix with shift info
        synthetic_suffix = f'sh{epoch_shift}'
        s3_uris = {
            synthetic_suffix: {
                'train': concat_uris['train'],
                'eval': concat_uris['eval'],
                'train_y': concat_uris['train_y'],
                'eval_y': concat_uris['eval_y']
            }
        }

        # Create WalletModeler with modified config
        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=modified_modeling_config,
            date_suffix=synthetic_suffix,
            s3_uris=s3_uris,
            override_approvals=self.wallets_config['workflow']['override_existing_models']
        )

        # Launch training
        logger.info(f"Starting training for epoch_shift={epoch_shift}")
        result = modeler.train_model()

        # Add epoch_shift info to result for tracking
        result['epoch_shift'] = epoch_shift
        result['synthetic_suffix'] = synthetic_suffix

        return result


    def train_all_models(self):
        """
        Train models for all loaded date suffixes using uploaded S3 data.

        Returns:
        - dict: Training results keyed by date suffix
        """
        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Call load_all_training_data() first.")

        # Get S3 URIs for all dates
        s3_uris = self.retrieve_training_data_uris(self.date_suffixes)

        training_results = {}
        n_threads = self.wallets_config['n_threads']['train_all_models']

        logger.info(f"Training models for {len(self.date_suffixes)} date periods with {n_threads} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_date = {
                executor.submit(self._train_single_model, date_suffix, s3_uris): date_suffix
                for date_suffix in self.date_suffixes
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date_suffix = future_to_date[future]
                result = future.result()
                training_results[date_suffix] = result

        logger.info(f"All {len(training_results)} models trained successfully.")
        return training_results


    def train_temporal_cv_model(self, date_suffixes: list[str]) -> dict:
        """
        Train a cross-date CV model: each date_suffix is treated as one fold.

        Workflow:
        1. Retrieve S3 URIs for all date_suffixes.
        2. Stage combined CV files to S3 via upload_temporal_cv_files().
        3. Launch a single script-mode CV job to train across folds.
        4. Return the training job result.

        Params:
        - date_suffixes (list[str]): List of date suffix strings.

        Returns:
        - dict: Contains model URI, training_job_name, and date_suffixes.
        """
        # 1. Get URIs for all dates
        s3_uris = self.retrieve_training_data_uris(date_suffixes)

        # 2. Upload temporal CV files and get root S3 URI
        cv_s3_uri = self.compile_temporal_cv_files(date_suffixes, s3_uris)

        # 3. Launch the CV training job
        result = sm.train_single_period_script_model(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            cv_s3_uri=cv_s3_uri,
            override_approvals=self.wallets_config['workflow']['override_existing_models']
        )

        # 4. Return the job metadata
        return {
            'model_uri': result.get('model_uri'),
            'training_job_name': result.get('training_job_name'),
            'date_suffixes': date_suffixes
        }


    def compile_temporal_cv_files(
            self,
            date_suffixes: list[str],
            s3_uris: dict[str, dict[str, str]]
        ) -> str:
        """
        Copy per-period train/eval CSVs into a cross-date CV folder in S3.

        Returns:
        - str: S3 URI of the root CV folder.
        """
        s3_client = boto3.client("s3")
        bucket = self.wallets_config['aws']['training_bucket']
        temporal_prefix = self.wallets_config['aws']['temporal_cv_directory']
        # Use a deterministic folder based on upload_directory
        upload_dir = self.wallets_config['training_data']['upload_directory']
        cv_prefix = f"{temporal_prefix}/{upload_dir}"

        # Check if CV folder already exists in S3
        existing = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=cv_prefix + '/'
        ).get('KeyCount', 0) > 0
        if existing:
            msg = f"s3://{bucket}/{cv_prefix}/ already exists. Overwrite existing CV files?"
            overwrite_all = u.request_confirmation(msg)
        else:
            overwrite_all = True

        uploaded_count = 0
        for suffix in date_suffixes:
            fold_prefix = f"{cv_prefix}/fold_{suffix}"
            # For train.csv
            src_train_uri = s3_uris[suffix]['train']
            src_train_key = src_train_uri.replace(f"s3://{bucket}/", "")
            dest_train_key = f"{fold_prefix}/train.csv"
            try:
                s3_client.head_object(Bucket=bucket, Key=dest_train_key)
                exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    exists = False
                else:
                    raise
            if not exists or overwrite_all:
                s3_client.copy({'Bucket': bucket, 'Key': src_train_key}, bucket, dest_train_key)
                uploaded_count += 1

            # For validation.csv
            src_eval_uri = s3_uris[suffix]['eval']
            src_eval_key = src_eval_uri.replace(f"s3://{bucket}/", "")
            dest_eval_key = f"{fold_prefix}/validation.csv"
            try:
                s3_client.head_object(Bucket=bucket, Key=dest_eval_key)
                exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    exists = False
                else:
                    raise
            if not exists or overwrite_all:
                s3_client.copy({'Bucket': bucket, 'Key': src_eval_key}, bucket, dest_eval_key)
                uploaded_count += 1

        logger.info(f"Uploaded {uploaded_count} CV files to s3://{bucket}/{cv_prefix}")
        # Return the S3 URI for the CV folder
        return f"s3://{bucket}/{cv_prefix}"


    def get_hpo_results(self, date_suffix: str = None) -> dict:
        """Get HPO results for a specific date_suffix."""
        if not date_suffix:
            date_suffix = self.date_suffixes[0] if self.date_suffixes else None
            if not date_suffix:
                raise ValueError("No date_suffix provided and none available")

        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=date_suffix
        )

        return modeler.load_hpo_results()


    @u.timing_decorator
    def predict_with_all_models(
            self,
            dataset_types: list = None,
            download_preds: bool = True
        ):
        """
        Generate predictions for test and val datasets across all date suffixes using their
         respective trained models.

        Uses nested concurrency: n_threads date suffixes × len(dataset_types) datasets.

        Params:
        - dataset_types (list): List of dataset types to predict (defaults to ['test', 'val'])
        - download_preds (bool): Whether to download predictions locally

        Returns:
        - dict: Prediction results keyed by date suffix {date_suffix: {dataset_type: result}}
        """
        if dataset_types is None:
            dataset_types = ['test', 'val']

        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Call load_all_training_data() first.")

        # Get S3 URIs for all dates
        s3_uris = self.retrieve_training_data_uris(self.date_suffixes)

        prediction_results = {}
        n_threads = self.wallets_config['n_threads']['predict_all_models']

        logger.milestone(f"Generating predictions for {len(self.date_suffixes)} date "
                         f"periods with {n_threads} threads...")
        logger.info(f"Dataset types: {dataset_types}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_date = {
                executor.submit(
                    self._predict_with_single_model,
                    date_suffix,
                    s3_uris,
                    dataset_types,
                    download_preds
                ): date_suffix
                for date_suffix in self.date_suffixes
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date_suffix = future_to_date[future]
                result = future.result()
                prediction_results[date_suffix] = result

        logger.milestone(f"All {len(prediction_results)} models generated predictions successfully.")
        return prediction_results


    @u.timing_decorator
    def train_concatenated_offsets_model(
            self,
            upload_results: dict,
            overwrite_existing: bool = False
        ) -> dict:
        """
        Build a full-history model by concatenating all offsets, uploading the
        combined CSVs, and training one XGBoost job on train/eval from that set.
        Returns the training job metadata.

        Params:
        - upload_results (dict): URIs from upload_concatenated_training_data()
            Format: {'train': s3://…/train.csv, 'eval': s3://…/eval.csv, 'test': …}
        """
        # Initialize date_suffixes for concatenation based on config offsets
        train_offsets = self.wallets_config['training_data'].get('train_offsets', [])
        eval_offsets  = self.wallets_config['training_data'].get('eval_offsets', [])
        test_offsets  = self.wallets_config['training_data'].get('test_offsets', [])
        # Use all offsets that have preprocessed CSVs available
        self.date_suffixes = list(dict.fromkeys(train_offsets + eval_offsets + test_offsets))

        # Prepare s3_uris dict keyed by our special suffix
        synthetic_suffix = 'concat'
        s3_uris = {
            synthetic_suffix: {
                'train': upload_results['train'],
                'eval':  upload_results['eval']
            }
        }
        # Always include y channel URIs
        s3_uris[synthetic_suffix]['train_y'] = upload_results['train_y']
        s3_uris[synthetic_suffix]['eval_y'] = upload_results['eval_y']

        # Launch training via WalletModeler
        modeler = WalletModeler(
            wallets_config   = self.wallets_config,
            modeling_config = self.modeling_config,
            date_suffix     = synthetic_suffix,
            s3_uris         = s3_uris,
            override_approvals = overwrite_existing
        )
        result = modeler.train_model()  # fits on train and early-stops on eval

        # 5) Return metadata for downstream (e.g. test scoring)
        return result


    def evaluate_all_models(self):
        """
        Run complete evaluation pipeline for all date suffixes using their respective predictions.

        Returns:
        - dict: Evaluation results keyed by date suffix {date_suffix: evaluator}
        """
        if not self.date_suffixes:
            raise ValueError("No date suffixes available. Call load_all_training_data() first.")

        evaluation_results = {}
        n_threads = self.wallets_config['n_threads']['evaluate_all_models']

        logger.milestone(f"Evaluating {len(self.date_suffixes)} models with {n_threads} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_date = {
                executor.submit(self._evaluate_single_model, date_suffix): date_suffix
                for date_suffix in self.date_suffixes
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date_suffix = future_to_date[future]
                evaluator = future.result()
                evaluation_results[date_suffix] = evaluator

        logger.milestone(f"All {len(evaluation_results)} model evaluations completed successfully.")
        return evaluation_results


    def _evaluate_single_model(self, date_suffix: str, log_summary: bool = True):
        """
        Run complete evaluation for a single date suffix model.

        Params:
        - date_suffix (str): Date suffix for this evaluation

        Returns:
        - RegressorEvaluator or ClassifierEvaluator: Completed evaluator instance
        """
        # Load predictions for this date suffix
        y_test_pred = sime.load_bt_sagemaker_predictions(
            'test',
            self.wallets_config,
            date_suffix
        )
        y_val_pred = sime.load_bt_sagemaker_predictions(
            'val',
            self.wallets_config,
            date_suffix
        )

        # Run complete evaluation pipeline
        evaluator = sime.create_sagemaker_evaluator(
            self.wallets_config,
            self.modeling_config,
            date_suffix,
            y_test_pred,
            y_val_pred
        )

        logger.info(f"Successfully generated evaluator for {date_suffix}.")

        if log_summary:
            evaluator.summary_report()

        return evaluator




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


    def _filter_temporal_overlap(self, date_data: dict, date_suffix: str) -> dict:
        """
        Filter each DataFrame to its max epoch_start_date to prevent temporal overlap.

        Params:
        - date_data (dict): Raw training data with keys like 'x_train', 'y_train', etc.
        - date_suffix (str): Date suffix for logging

        Returns:
        - dict: Filtered training data with same structure
        """
        filtered_data = {}
        total_rows_removed = 0
        original_rows = 0

        for key, df in date_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                filtered_data[key] = df
                continue

            # Get max date for this DataFrame and filter to only that date
            max_date = df.index.get_level_values('epoch_start_date').max()
            mask = df.index.get_level_values('epoch_start_date') == max_date
            filtered_data[key] = df[mask]

            # Track filtering stats
            rows_removed = len(df) - len(filtered_data[key])
            total_rows_removed += rows_removed
            original_rows += len(df)

        # Log filtering results
        if total_rows_removed > 0:
            logger.info(f"Temporal filtering for {date_suffix}: removed {total_rows_removed:,} rows "
                        f"({total_rows_removed/original_rows*100:.1f}%) to prevent overlap")
        else:
            logger.debug(f"Temporal filtering for {date_suffix}: no rows removed")

        return filtered_data


    def _load_preprocessed_training_data(self, date_suffixes: list) -> dict:
        """
        Load preprocessed CSV files for upload, ensuring perfect consistency
        between saved files and uploaded data.
        """
        # Get the preprocessed data directory from SageWalletsPreprocessor logic
        base_dir = (Path(f"{self.wallets_config['training_data']['local_s3_root']}")
                    / "s3_uploads"
                    / "wallet_training_data_preprocessed")
        local_dir = self.wallets_config["training_data"]["local_directory"]
        if self.dataset == 'dev':
            local_dir = f"{local_dir}_dev"
        preprocessed_dir = base_dir / local_dir

        splits = ['train', 'test', 'eval', 'val']
        data_by_date = {}

        for date_suffix in date_suffixes:
            date_data = {}

            # Load CSV files from date-specific folder
            date_folder = preprocessed_dir / date_suffix
            if not date_folder.exists():
                raise FileNotFoundError(f"Date folder not found: {date_folder}")

            for split_name in splits:
                filename = f"{split_name}.csv"  # Changed from {split_name}_preprocessed_{date_suffix}.csv
                filepath = date_folder / filename

                if not filepath.exists():
                    raise FileNotFoundError(f"Preprocessed file not found: {filepath}")

                date_data[split_name] = pd.read_csv(filepath, header=None)

            # Load metadata from the same folder
            metadata_file = date_folder / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    date_data['metadata'] = json.load(f)
            else:
                raise FileNotFoundError(f"No metadata.json found in {date_folder}")

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
                # Always load full y files for target
                if data_type == 'y':
                    pattern = f"y_{split}_full_{date_suffix}.parquet"
                else:
                    pattern = f"{data_type}_{split}_{date_suffix}.parquet"

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


    def _upload_single_date(
            self,
            date_suffix: str,
            preprocessed_data: dict,
            context: UploadContext
        ) -> dict:
        """
        Coordinates the upload of all files for a single date suffix to S3.

        Params:
        - date_suffix (str): Date suffix for this upload
        - preprocessed_data (dict): Preprocessed data for this date
        - context (UploadContext): Upload configuration and metadata

        Returns:
        - dict: S3 URIs for all uploaded files for this date
        """
        s3_client = boto3.client('s3')

        # Upload CSV splits for this date
        date_uris = self._upload_csv_files(
            date_suffix,
            preprocessed_data,
            context,
            s3_client
        )

        # Upload metadata for this date
        metadata_uri = self._upload_metadata_for_date(
            date_suffix,
            preprocessed_data['metadata'],
            context,
            s3_client
        )
        date_uris['metadata'] = metadata_uri

        logger.info(f"Successfully completed upload for {date_suffix}")
        return date_uris



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


    def _validate_custom_filters_config(self, metadata_file: Path):
        """
        Validate custom filter configuration against feature metadata.

        Params:
        - metadata_file (Path): Path to the metadata.json file
        """
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        feature_columns = metadata['feature_columns']

        # Get filter definitions
        custom_filters = self.modeling_config['training'].get('custom_filters', {})

        if not custom_filters:
            logger.warning("No custom_filters defined, skipping config validation...")
            return

        # Validate filter keys exist in feature list
        missing_columns = []
        for filter_col in custom_filters.keys():
            if filter_col not in feature_columns:
                missing_columns.append(filter_col)

        if missing_columns:
            raise ValueError(f"Filter columns not found in features: {missing_columns}")

        # Validate filter values are numeric
        for filter_col, filter_rules in custom_filters.items():
            for rule_type, rule_value in filter_rules.items():
                if rule_type in ['min', 'max'] and not isinstance(rule_value, (int, float)):
                    raise ValueError(f"Filter value must be numeric: {filter_col}.{rule_type} "
                                     f"= {rule_value} ({type(rule_value)})")

        logger.info(f"Custom filter validation passed: {len(custom_filters)} filters validated")


    def _get_s3_upload_paths(self) -> tuple[str, str, str]:
        """
        Get S3 bucket, base folder, and upload folder prefix for training data.
        Automatically switches between preprocessed and concatenated directories
        based on concatenate_offsets config flag.

        Returns:
        - tuple: (bucket_name, base_folder, folder_prefix)
        """
        bucket_name = self.wallets_config['aws']['training_bucket']

        # Check if we should use concatenated directory instead of preprocessed
        use_concatenated = self.wallets_config['training_data'].get('concatenate_offsets', False)

        if use_concatenated:
            base_folder = self.wallets_config['aws']['concatenated_directory']
        else:
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
        """
        logger.milestone(
            f"<{context.dataset.upper()}> Ready to upload "
            f"{context.total_rows} total rows "
            f"({context.total_size_mb:.1f} MB) of preprocessed training data "
            f"across {len(self.date_suffixes)} date folders."
        )

        # If overwrite allowed, check for existing files
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

        # Log upload information and request approval WITH SIZE
        logger.info(
            f"Target: s3://{context.bucket_name}/"
            f"{context.base_folder}/{context.folder_prefix}[DATE]/"
        )
        confirmation = u.request_confirmation(f"Proceed with upload of {context.total_size_mb:.1f} "
                                              "MB? (y/N): ")
        if not confirmation:
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

            # Store df to temp CSV
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                df.to_csv(tmp.name, index=False, header=False)
                temp_path = tmp.name

            # Upload temp CSV
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


    def _train_single_model(self, date_suffix: str, s3_uris: dict) -> dict:
        """
        Train a model for a specific date suffix.

        Params:
        - date_suffix (str): Date suffix for this training run
        - s3_uris (dict): S3 URIs for all date suffixes

        Returns:
        - dict: Training results for this date suffix
        """
        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=date_suffix,
            s3_uris={date_suffix: s3_uris[date_suffix]},
            override_approvals= self.wallets_config['workflow']['override_existing_models']
        )

        result = modeler.train_model()
        logger.info(f"Successfully completed training for {date_suffix}")
        return result


    def _predict_with_single_model(
            self,
            date_suffix: str,
            s3_uris: dict,
            dataset_types: list = None,
            download_preds: bool = True
        ) -> dict:
        """
        Generate predictions for test and val datasets using a specific date suffix's trained model.

        Params:
        - date_suffix (str): Date suffix for this prediction run
        - s3_uris (dict): S3 URIs for all date suffixes
        - dataset_types (list): List of dataset types to predict (defaults to ['test', 'val'])
        - download_preds (bool): Whether to download predictions locally

        Returns:
        - dict: Prediction results for this date suffix {dataset_type: result}
        """
        if dataset_types is None:
            dataset_types = ['test', 'val']

        # Create WalletModeler instance for this date
        # Date format validation handled in WalletModeler.__init__()
        # Missing S3 URIs will be caught in predict_with_batch_transform() → _validate_s3_uris()
        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=date_suffix,
            s3_uris={date_suffix: s3_uris[date_suffix]},
            override_approvals=self.wallets_config['workflow']['override_existing_models']
        )

        # Load existing trained model - FileNotFoundError raised if no model exists
        model_info = modeler.load_existing_model()
        logger.debug(f"Loaded model for {date_suffix}: {model_info['training_job_name']}")

        # Generate predictions for datasets concurrently
        prediction_results = {}
        n_threads = self.wallets_config['n_threads']['predict_datasets']

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_dataset = {
                executor.submit(
                    modeler.predict_with_batch_transform,
                    dataset_type,
                    download_preds,
                    dataset_type  # Pass dataset_type as job_name_suffix for unique job names
                ): dataset_type
                for dataset_type in dataset_types
            }

            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_type = future_to_dataset[future]
                result = future.result()
                prediction_results[dataset_type] = result
                logger.debug(f"Successfully generated predictions for {dataset_type} on {date_suffix}")

        logger.milestone(f"Successfully completed predictions for {date_suffix}: {list(dataset_types)}")
        return prediction_results


# ---------------------------------
#         Utility Functions
# ---------------------------------

def calculate_comprehensive_offsets_by_split(wallets_config: dict) -> dict[str, list[str]]:
    """
    Calculate date suffixes needed for each split across all epoch shifts.

    Params:
    - wallets_config (dict): Wallets configuration

    Returns:
    - dict: Split-specific date requirements
        {
            'all_train_offsets': ['220615', '220715', ...],
            'all_eval_offsets': ['230809', '230908', ...],
            'all_test_offsets': ['230809', '230908', ...],
            'all_val_offsets': ['240405', '240505', ...]
        }
    """
    epoch_shifts = wallets_config['training_data'].get('epoch_shifts', [0])
    base_splits = ['train_offsets', 'eval_offsets', 'test_offsets', 'val_offsets']

    result = {}

    for split_name in base_splits:
        if split_name not in wallets_config['training_data']:
            result[f'all_{split_name}'] = []
            continue

        split_dates = set()

        # For each shift, get the shifted dates for this specific split
        for shift in epoch_shifts:
            shifted_offset_ints = ct.identify_offset_ints(wallets_config, shift=shift)
            split_key = split_name  # 'train_offsets', 'eval_offsets', etc.

            if split_key in shifted_offset_ints:
                shifted_dates = convert_offset_ints_to_dates(
                    shifted_offset_ints[split_key],
                    wallets_config
                )
                split_dates.update(shifted_dates)

        result[f'all_{split_name}'] = sorted(list(split_dates))

    return result


def convert_offset_ints_to_dates(offset_days: list[int], wallets_config: dict) -> list[str]:
    """
    Convert integer days since date_0 back to YYMMDD date strings.

    Params:
    - offset_days (list[int]): List of days since date_0 (e.g., [30, 60, 90])
    - wallets_config (dict): Contains date_0 reference date

    Returns:
    - list[str]: YYMMDD date strings (e.g., ['200226', '200327', '200426'])
    """
    # Parse reference date
    date_0_str = str(wallets_config['training_data']['date_0'])
    date_0 = pd.to_datetime(date_0_str, format='%y%m%d')

    date_strings = []
    for days in offset_days:
        target_date = date_0 + pd.Timedelta(days=days)
        # Convert back to YYMMDD format
        date_string = target_date.strftime('%y%m%d')
        date_strings.append(date_string)

    return date_strings
