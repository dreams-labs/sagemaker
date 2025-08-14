"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
import copy
import traceback
import tempfile
import tarfile
import time
from typing import List
from pathlib import Path
import concurrent.futures
import json
import numpy as np
import pandas as pd
import boto3
import xgboost as xgb
from botocore.exceptions import ClientError

# Local modules
from sage_wallet_modeling.wallet_preprocessor import SageWalletsPreprocessor
from sage_wallet_modeling.wallet_modeler import WalletModeler
# For cross-date CV training
import script_modeling.custom_transforms as ct
import utils as u
from utils import ConfigError
import sage_utils.config_validation as ucv
from sage_wallet_insights import model_evaluation as sime

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


    @u.timing_decorator
    def preprocess_all_training_data(self):
        """
        Preprocess training data for all loaded date suffixes independently.

        If concatenate_offsets is enabled, filters each date_suffix to prevent
        temporal overlap during downstream concatenation.
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_all_training_data() first.")

        preprocessed_by_date = {}

        logger.info(f"Preprocessing {len(self.training_data)} date periods...")

        for date_suffix, date_data in self.training_data.items():
            logger.debug(f"Preprocessing data for {date_suffix}...")

            # Apply temporal filtering to prevent overlap
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
    def concatenate_all_preprocessed_data(self) -> None:
        """
        Concatenate preprocessed Parquet files across configured offsets for each split,
        save combined feature CSVs (no headers/indices) to the concatenated output
        directory, and also save the concatenated index (only) as a Parquet sidecar.

        Exports separate y-files for test and val splits using RAW (non-preprocessed)
        y values for evaluation purposes.
        """
        logger.info("Loading preprocessed training data...")
        data_by_date = self._load_preprocessed_training_data(self.date_suffixes)

        logger.info("Beginning concatenation of preprocessed data from Parquet...")
        # Determine offsets for each split from config
        split_requirements = calculate_comprehensive_offsets_by_split(self.wallets_config)

        offsets_map = {
            'train': split_requirements['all_train_offsets'],
            'eval':  split_requirements['all_eval_offsets'],
            'test':  split_requirements['all_test_offsets'],
            'val':   split_requirements['all_val_offsets']
        }
        # Preflight: ensure no NaNs and consistent column counts across all date_suffixes per split
        for split, offsets in offsets_map.items():
            if not offsets:
                continue
            ncols_by_date = {}
            for offset in offsets:
                if offset not in data_by_date or split not in data_by_date[offset]:
                    raise KeyError(f"Missing data for offset '{offset}' split '{split}'")
                df_loaded = data_by_date[offset][split]
                if df_loaded.isnull().values.any():
                    bad_cols = df_loaded.columns[df_loaded.isnull().any()].tolist()
                    raise ValueError(f"NaNs detected in loaded preprocessed Parquet for "
                                     f"{offset}/{split}: columns {bad_cols}")
                ncols_by_date[offset] = df_loaded.shape[1]
            if len(set(ncols_by_date.values())) != 1:
                raise ValueError(f"Column count mismatch for split '{split}' across dates: {ncols_by_date}")

        # Build concatenation output directory alongside the preprocessed tree
        base_dir = Path(self.wallets_config['training_data']['local_s3_root']) \
                / "s3_uploads" \
                / "wallet_training_data_concatenated"
        local_dir = self.wallets_config["training_data"]["local_directory"]
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

            # Concatenate preserving index (multi-index expected) and validate
            concatenated = pd.concat(dfs, axis=0)
            if concatenated.isnull().any().any():
                raise ValueError(f"NaN values detected in {split} split before saving to CSV")

            # Write index-only sidecar as Parquet (row-aligned with features)
            index_out_file = concat_base / f"{split}_index.parquet"
            index_df = concatenated.index.to_frame(index=False)
            index_df.to_parquet(index_out_file, index=False)
            logger.info(
                f"Saved concatenated {split}_index.parquet with {len(index_df)} rows to {index_out_file}"
            )

            # Write features as CSV.gz (no header, no index) for SageMaker compatibility
            out_file = concat_base / f"{split}.csv.gz"
            concatenated.to_csv(out_file, index=False, header=False, compression='gzip')
            logger.info(
                f"Saved concatenated {split}.csv.gz with {len(concatenated)} rows to {out_file}"
            )

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
            concatenated_y.to_csv(y_out_file, index=False, header=True)
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
        folder_prefix = f"{upload_directory}/"

        s3_client = boto3.client('s3')
        upload_results = {}

        # Local concatenated directory
        concat_root = Path(self.wallets_config['training_data']['local_s3_root']) \
                      / "s3_uploads" \
                      / "wallet_training_data_concatenated"
        local_dir = self.wallets_config["training_data"]["local_directory"]
        concat_dir = concat_root / local_dir

        logger.info(f"Beginning upload of concatenated training data for splits {splits}...")
        # Parallel upload of concatenated splits
        n_threads = self.wallets_config['n_threads']['upload_all_training_data']
        logger.info(f"Uploading concatenated splits in parallel with {n_threads} threads...")
        def _upload_split(split: str):
            is_y = split.endswith('_y')
            filename = f"{split}.csv" if is_y else f"{split}.csv.gz"
            s3_key = f"{base_folder}/{folder_prefix}{filename}"
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
            local_file = concat_dir / filename
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


    def train_all_epoch_shift_models(
        self,
        concat_uris: dict[str, str],
        retry_attempts: int = 3,
        retry_delay_seconds: int = 30,
        retry_backoff: float = 2.0,
    ) -> dict[int, dict]:
        """
        Train models for all epoch shifts using concatenated data with temporal filtering.

        Each model uses the same master concatenated dataset but applies different
        epoch_shift filtering in the container to train on different temporal windows.

        Params:
        - epoch_shifts (list[int]): List of shift values (e.g., [0, 30, 60, 90, 120])
        - concat_uris (dict): S3 URIs for concatenated data splits
            {'train': s3://..., 'eval': s3://..., 'train_y': s3://..., 'eval_y': s3://...}
        - retry_attempts (int): Number of times to retry if throttled.
        - retry_delay_seconds (int): Initial delay (in seconds) before the first retry.
        - retry_backoff (float): Multiplier applied to delay after each failed attempt (exponential backoff).

        Returns:
        - dict: Training results keyed by epoch_shift {shift: training_result}
            Only successfully executed shifts are returned; failures after retries include
            'error', 'type', and 'traceback'.
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
                    self._train_single_epoch_shift_with_retry,
                    shift,
                    concat_uris,
                    retry_attempts,
                    retry_delay_seconds,
                    retry_backoff,
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
                    logger.exception(f"Training failed for epoch_shift={shift}")
                    training_results[shift] = {
                        'error': repr(e),
                        'type': type(e).__name__,
                        'traceback': traceback.format_exc(),
                    }

        successful_models = len([r for r in training_results.values() if 'error' not in r])
        logger.milestone(f"Epoch shift training complete: {successful_models}/{len(epoch_shifts)} models successful")

        return training_results


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

        # Using concatenated directory structure

        # Get S3 file locations
        bucket_name, base_folder, folder_prefix = self._get_s3_upload_paths()

        s3_client = boto3.client('s3')
        s3_uris = {}
        splits = ['train', 'test', 'eval', 'val']

        for date_suffix in date_suffixes:
            date_uris = {}

            for split_name in splits:
                # Concatenated structure: files directly under upload directory
                # Path: s3://bucket/concatenated/{upload_dir}/train.csv
                prefix = f"{base_folder}/{folder_prefix}{split_name}"
                expected_filename = f"{split_name}.csv.gz"

                try:
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=prefix
                    )

                    if 'Contents' not in response:
                        logger.warning("No S3 objects found matching prefix: "
                                       f"s3://{bucket_name}/{prefix}")
                        continue

                    # Look for exact filename match in concatenated layout
                    matching_objects = [
                        obj for obj in response['Contents']
                        if obj['Key'].split('/')[-1] == expected_filename
                    ]

                    if len(matching_objects) == 0:
                        raise FileNotFoundError(
                            f"No concatenated CSV file found for '{split_name}' at: s3://{bucket_name}/{prefix}"
                        )

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


    def predict_all_epoch_shifts(
            self,
            overwrite_existing: bool = False,
            retry_attempts: int = 3,
            retry_delay_seconds: int = 30,
            retry_backoff: float = 2.0
        ) -> dict[int, dict]:
        """
        Generate predictions for all epoch shift models using concatenated test/val data.
        Only run predictions if local prediction files do not already exist, unless
        overwrite_existing=True.

        Params:
        - overwrite_existing (bool): If False, skip epoch_shifts where both local
          predictions exist. If True, re-run and overwrite any existing predictions.
        - retry_attempts (int): Number of times to retry if throttled.
        - retry_delay_seconds (int): Initial delay (in seconds) before the first retry.
        - retry_backoff (float): Multiplier applied to delay after each failed attempt (exponential backoff).

        Returns:
        - dict: {epoch_shift: {'test': result, 'val': result, 'model_uri': str}} for shifts executed.
            Only successfully executed shifts are returned. Failures after retries are included
            with an 'error' key as before.
        """
        epoch_shifts = self.wallets_config['training_data']['epoch_shifts']
        n_threads = self.wallets_config['n_threads']['predict_all_models']

        if not epoch_shifts:
            raise ConfigError("No epoch_shifts configured in wallets_config")

        # Load concatenated data URIs once
        concat_uris = self.retrieve_training_data_uris(['concat'])

        # Determine which shifts actually need work
        shifts_to_run: list[int] = []
        for epoch_shift in epoch_shifts:
            if not overwrite_existing and self._local_predictions_exist(epoch_shift):
                logger.info(
                    f"Skipping epoch_shift={epoch_shift}: local predictions already exist "
                    f"and overwrite_existing=False"
                )
                continue
            shifts_to_run.append(epoch_shift)

        if not shifts_to_run:
            logger.milestone("Epoch shift predictions skipped: all predictions exist locally.")
            return {}

        logger.milestone(
            f"Starting predictions for {len(shifts_to_run)} epoch shifts using {n_threads} threads..."
        )

        prediction_results: dict[int, dict] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit only required epoch shift predictions
            future_to_shift = {
                executor.submit(
                    self._predict_single_epoch_shift_with_retry,
                    epoch_shift,
                    concat_uris,
                    overwrite_existing,
                    retry_attempts,
                    retry_delay_seconds,
                    retry_backoff,
                ): epoch_shift
                for epoch_shift in shifts_to_run
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_shift):
                shift = future_to_shift[future]
                try:
                    result = future.result()
                    prediction_results[shift] = result
                    logger.milestone(f"✓ Completed predictions for epoch_shift={shift}")
                except Exception as e:
                    logger.error(f"✗ Failed predictions for epoch_shift={shift}: {e}")
                    prediction_results[shift] = {'error': str(e)}

        # Summary
        successful = len([r for r in prediction_results.values() if 'error' not in r])
        logger.milestone(f"Epoch shift predictions complete: {successful}/{len(prediction_results)} successful")

        return prediction_results


    @u.timing_decorator
    def build_all_epoch_shift_evaluators(self) -> dict[int, object]:
        """
        Build a model evaluator for each configured epoch shift using concatenated
        test/val data and return them in a dict keyed by epoch_shift.

        Returns:
        - dict: {epoch_shift: evaluator or {'error': str}}
        """
        epoch_shifts = self.wallets_config['training_data'].get('epoch_shifts', [])
        if not epoch_shifts:
            raise ConfigError("No epoch_shifts configured in wallets_config")

        n_threads = self.wallets_config['n_threads']['evaluate_all_models']
        logger.milestone(
            f"Building evaluators for {len(epoch_shifts)} epoch shifts using {n_threads} threads..."
        )

        evaluators_by_shift: dict[int, object] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_shift = {
                executor.submit(self._build_evaluator_for_shift, shift): shift
                for shift in epoch_shifts
            }

            for future in concurrent.futures.as_completed(future_to_shift):
                shift = future_to_shift[future]
                try:
                    evaluator = future.result()
                    evaluators_by_shift[shift] = evaluator
                    logger.milestone(f"✓ Built evaluator for epoch_shift={shift}")
                except Exception as e:
                    logger.error(f"✗ Failed to build evaluator for epoch_shift={shift}: {e}")
                    evaluators_by_shift[shift] = {'error': str(e)}
                    raise e

        return dict(sorted(evaluators_by_shift.items()))









    # ------------------------
    #      Helper Methods
    # ------------------------

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
        Load preprocessed Parquet files for upload, ensuring perfect consistency
        between saved files and uploaded data.
        """
        # Get the preprocessed data directory from SageWalletsPreprocessor logic
        base_dir = (Path(f"{self.wallets_config['training_data']['local_s3_root']}")
                    / "s3_uploads"
                    / "wallet_training_data_preprocessed")
        local_dir = self.wallets_config["training_data"]["local_directory"]
        preprocessed_dir = base_dir / local_dir

        splits = ['train', 'test', 'eval', 'val']
        data_by_date = {}

        for date_suffix in date_suffixes:
            date_data = {}

            # Load Parquet files from date-specific folder (index preserved)
            date_folder = preprocessed_dir / date_suffix
            if not date_folder.exists():
                raise FileNotFoundError(f"Date folder not found: {date_folder}")

            for split_name in splits:
                # Load Parquet files from date-specific folder (index preserved)
                parquet_path = date_folder / f"{split_name}.parquet"
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Preprocessed file not found: {parquet_path}")

                date_data[split_name] = pd.read_parquet(parquet_path)
                # Validate: no NaNs in any loaded preprocessed Parquet
                if date_data[split_name].isnull().values.any():
                    bad_cols = date_data[split_name].columns[date_data[split_name].isnull().any()].tolist()
                    logger.error(
                        "NaNs found in preprocessed Parquet for %s/%s; columns with NaNs: %s",
                        date_suffix, split_name, bad_cols
                    )
                    raise ValueError(
                        f"NaNs found in preprocessed Parquet for {date_suffix}/{split_name}: columns {bad_cols}"
                    )

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


    def _train_single_epoch_shift_with_retry(
        self,
        epoch_shift: int,
        concat_uris: dict[str, str],
        retry_attempts: int,
        retry_delay_seconds: int,
        retry_backoff: float,
    ) -> dict:
        """Call `_train_single_epoch_shift` with throttling-aware retries."""
        attempt = 1
        delay = max(0, int(retry_delay_seconds))
        last_exc = None
        max_attempts = max(1, int(retry_attempts))
        while attempt <= max_attempts:
            try:
                return self._train_single_epoch_shift(epoch_shift, concat_uris)
            except Exception as exc:  # noqa: BLE001
                if _is_throttling_error(exc) and attempt < max_attempts:
                    logger.warning(
                        "Throttled on training epoch_shift=%s (attempt %s/%s). Sleeping %ss then retrying... Error: %s",
                        epoch_shift,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    try:
                        delay = int(delay * float(retry_backoff)) if retry_backoff else delay
                    except Exception:
                        pass
                    attempt += 1
                    last_exc = exc
                    continue
                last_exc = exc
                break
        if last_exc is not None:
            raise last_exc
        return {}


    def _get_s3_upload_paths(self) -> tuple[str, str, str]:
        """
        Get S3 bucket, base folder, and upload folder prefix for training data.
        Automatically switches between preprocessed and concatenated directories
        based on concatenate_offsets config flag.

        Returns:
        - tuple: (bucket_name, base_folder, folder_prefix)
        """
        bucket_name = self.wallets_config['aws']['training_bucket']

        # Always use concatenated directory
        base_folder = self.wallets_config['aws']['concatenated_directory']

        upload_directory = self.wallets_config['training_data']['upload_directory']
        folder_prefix = f"{upload_directory}/"

        return bucket_name, base_folder, folder_prefix


    def _local_predictions_exist(self, epoch_shift: int) -> bool:
        """
        Check whether both test and val local prediction files already exist and are non-empty
        for a given epoch_shift (sh{epoch_shift}).

        Returns:
        - bool: True if both files exist and have size > 0.
        """
        local_root = (
            Path(self.wallets_config['training_data']['local_s3_root'])
            / "s3_downloads"
            / "wallet_predictions"
            / self.wallets_config['training_data']['local_directory']
            / f"sh{epoch_shift}"
        )
        test_path = local_root / "test.csv.out"
        val_path = local_root / "val.csv.out"

        try:
            return (
                test_path.exists() and test_path.stat().st_size > 0 and
                val_path.exists() and val_path.stat().st_size > 0
            )
        except OSError:
            # If stat() fails for any reason, treat as non-existent to be safe
            return False


    def _predict_single_epoch_shift(
            self,
            epoch_shift: int,
            concat_uris: dict,
            overwrite_existing: bool = False
        ) -> dict:
        """
        Helper to predict test and val for a single epoch shift.

        Params:
        - epoch_shift (int): The epoch shift value
        - concat_uris (dict): S3 URIs for concatenated data
        - overwrite_existing (bool): Whether to overwrite existing local predictions

        Returns:
        - dict: Prediction results for this epoch shift
        """
        # Create modeler with epoch_shift as the date_suffix
        shift_s3_uris = {f'sh{epoch_shift}': concat_uris['concat']}

        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=f'sh{epoch_shift}',
            s3_uris=shift_s3_uris,
            override_approvals=True  # Skip confirmations in parallel execution
        )

        # Load the model for this epoch_shift
        try:
            model_info = modeler.load_existing_model(epoch_shift=epoch_shift)
            logger.debug(f"Loaded model for epoch_shift={epoch_shift}: {model_info['model_uri']}")
        except FileNotFoundError as e:
            logger.error(f"No model found for epoch_shift={epoch_shift}")
            raise e

        # Run batch predictions for test and val with epoch filtering
        pred_results = modeler.batch_predict_test_and_val(
            overwrite_existing=overwrite_existing,
            offset_filters=None
        )

        # Compile results
        return {
            'test': pred_results['test'],
            'val': pred_results['val'],
            'model_uri': model_info['model_uri'],
            'training_job_name': model_info.get('training_job_name'),
            'epoch_shift': epoch_shift
        }


    def _predict_single_epoch_shift_with_retry(
        self,
        epoch_shift: int,
        concat_uris: dict[str, str],
        overwrite_existing: bool,
        retry_attempts: int,
        retry_delay_seconds: int,
        retry_backoff: float,
    ) -> dict:
        """Call `_predict_single_epoch_shift` with throttling-aware retries."""
        attempt = 1
        delay = max(0, int(retry_delay_seconds))
        last_exc = None
        while attempt <= max(1, int(retry_attempts)):
            try:
                return self._predict_single_epoch_shift(epoch_shift, concat_uris, overwrite_existing)
            except Exception as exc:  # noqa: BLE001 - we intentionally catch broadly to detect throttling
                if _is_throttling_error(exc) and attempt < max(1, int(retry_attempts)):
                    logger.warning(
                        "Throttled on epoch_shift=%s (attempt %s/%s). Sleeping %ss then retrying... Error: %s",
                        epoch_shift,
                        attempt,
                        retry_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    # Exponential backoff for next attempt
                    try:
                        delay = int(delay * float(retry_backoff)) if retry_backoff else delay
                    except Exception:
                        # If backoff isn't a valid float, keep the same delay
                        pass
                    attempt += 1
                    last_exc = exc
                    continue
                # Non-throttling error or final attempt exhausted
                last_exc = exc
                break
        # If we reach here, all attempts failed
        if last_exc is not None:
            raise last_exc
        # Should never get here, but return {} to satisfy type checker
        return {}


    def _build_evaluator_for_shift(self, epoch_shift: int):
        """
        Build a single evaluator for the provided epoch shift.

        Steps:
        - Load model info (to obtain model_uri)
        - Load batch transform predictions for test/val; if missing, run batch transform
        - Load concatenated y for test/val and align target column name
        - Construct and return the evaluator object
        """
        date_suffix = f"sh{epoch_shift}"

        # Initialize a modeler to get model info and (if needed) run predictions
        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=date_suffix,
            s3_uris=None,
            override_approvals=True
        )

        # Load model info (raises if not found)
        model_info = modeler.load_existing_model(epoch_shift=epoch_shift)

        # Load concatenated y for and y_pred
        y_test = sime.load_concatenated_y('test', self.wallets_config, self.modeling_config)
        y_val  = sime.load_concatenated_y('val',  self.wallets_config, self.modeling_config)
        y_test_pred = sime.load_bt_sagemaker_predictions('test', self.wallets_config, date_suffix)
        y_val_pred  = sime.load_bt_sagemaker_predictions('val',  self.wallets_config, date_suffix)

        target_var = self.modeling_config['target']['target_var']
        y_test.columns = [target_var]
        y_val.columns = [target_var]

        # Build the evaluator using the shared helper in sage_wallet_insights
        evaluator = sime.create_concatenated_sagemaker_evaluator(
            self.wallets_config,
            self.modeling_config,
            model_info['model_uri'],
            y_test_pred,
            y_test,
            y_val_pred,
            y_val,
            epoch_shift
        )

        return evaluator


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


def _is_throttling_error(exc: Exception) -> bool:
    """Return True if the exception appears to be an AWS throttling error."""
    # Botocore ClientError:
    if isinstance(exc, ClientError):
        code = exc.response.get('Error', {}).get('Code', '')
        message = exc.response.get('Error', {}).get('Message', '')
        if code in {"ThrottlingException", "Throttling", "TooManyRequestsException", "RequestLimitExceeded"}:
            return True
        if isinstance(message, str) and ("Rate exceeded" in message or "Throttling" in message):
            return True
    # Fallback string check for other exceptions (e.g., SDK wrappers)
    text = str(exc)
    return ("Rate exceeded" in text) or ("Throttling" in text)
