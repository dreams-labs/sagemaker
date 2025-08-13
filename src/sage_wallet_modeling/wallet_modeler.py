"""
Class to manage all steps of the wallet model training and scoring.

This class handles data that has already been feature engineered and uploaded to S3,
 indexed on a wallet-coin-offset_date tuple, with features already present as columns.

Interacts with:
---------------
WalletWorkflowOrchestrator: uses this class for model construction
"""
import logging
from typing import Dict,Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Local module imports
import utils as u
from utils import ConfigError
import sage_utils.config_validation as ucv
import sage_utils.s3_utils as s3u
import sage_wallet_modeling.wallet_script_modeler as sm


# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletModeler:
    """
    Handles model training and prediction generation for wallet-coin performance modeling
     using SageMaker with S3 data sources. Manages SageMaker infrastructure configuration,
     hyperparameter settings, and model artifacts for wallet performance prediction workflows.

    The modeling period is specified using the date_suffix param. Each WalletModeler is
     responsible for generating the specified modeling period's predictions using
     that modeling period's training data.

    Params:
    - wallets_config (dict): abbreviated name for sage_wallets_config.yaml
    - modeling_config (dict): abbreviated name for sage_wallets_modeling_config.yaml
    - date_suffix (str): the modeling_period_start of the training data with which to
        build the model.
    - s3_uris (dict): dict containing S3 URIs locating the training data CSV files for
        each date_suffix, formatted as:
            {date_suffix}:
                train: {uri}
                test:  {uri}
                eval:  {uri}
                val:   {uri}
    - override_approvals (Optional[bool]): if None, interactive confirmations; if True or False,
        override confirmations with that value.
    """
    def __init__(
            self,
            wallets_config: Dict,
            modeling_config: Dict,
            date_suffix: str,
            s3_uris: Dict[str, Dict[str, str]] = None,
            override_approvals: Optional[bool] = None
        ):
        # Configs
        ucv.validate_sage_wallets_config(wallets_config)
        ucv.validate_sage_wallets_modeling_config(modeling_config)
        self.wallets_config = wallets_config
        self.modeling_config = modeling_config

        # SageMaker setup
        self.sagemaker_session = sagemaker.Session()
        self.role = wallets_config['aws']['modeler_arn']
        self.s3_uris = s3_uris
        self.date_suffix = date_suffix

        # Store upload folder as instance state
        self.upload_directory = wallets_config['training_data']['upload_directory']

        # Model artifacts
        self.model_uri = None
        self.predictions_uri = None
        self.endpoint_name: Optional[str] = None
        self.predictor: Optional[sagemaker.predictor.Predictor] = None

        # Misc
        self.override_approvals = override_approvals


    # ------------------------
    #      Public Methods
    # ------------------------

    def train_model(self):
        """
        Train model using SageMaker's built-in algorithm.
         Uses train/test splits with eval for early stopping.

        Returns:
        - dict: Contains model URI and training job name
        """
        # If script-mode is enabled in config, delegate to the script-mode launcher
        return sm.initiate_script_modeling(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=self.date_suffix,
            s3_uris=self.s3_uris,
        )


    def load_existing_model(self, model_uri: str = None, epoch_shift: int = None):
        """
        Load the most recent trained model for a given date_suffix or epoch_shift.
        Handles script-mode model storage pattern.

        Params:
        - model_uri (str, optional): Specific model URI to load directly
        - epoch_shift (int, optional): Find model for specific epoch shift (uses sh{epoch_shift} pattern)

        Returns:
        - dict: Contains model URI and training job name of most recent model
        """
        # If a specific model_uri is provided, validate and load it directly
        if model_uri:
            if not model_uri.startswith('s3://'):
                raise ValueError(f"Invalid S3 URI format: {model_uri}")
            # Parse bucket and key
            _bucket, _key = model_uri.replace('s3://', '', 1).split('/', 1)
            _s3_client = self.sagemaker_session.boto_session.client('s3')
            try:
                _s3_client.head_object(Bucket=_bucket, Key=_key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise FileNotFoundError(f"Model file not found at specified model_uri: {model_uri}") from e
                else:
                    raise
            # Store and return
            self.model_uri = model_uri
            logger.info(f"Loaded model from specified model_uri: {model_uri}")
            return {
                'model_uri': model_uri,
                'training_job_name': None,
                'timestamp': None
            }

        # Determine effective date_suffix and job name pattern based on epoch_shift
        if epoch_shift is not None:
            effective_date_suffix = f"sh{epoch_shift}"
            job_name_pattern = f"wscr-{self.upload_directory[:8]}-sh{epoch_shift}-"
            logger.debug(f"Loading model for epoch_shift={epoch_shift} using pattern: {job_name_pattern}")
        else:
            effective_date_suffix = self.date_suffix
            job_name_pattern = f"wscr-{self.upload_directory[:8]}-{self.date_suffix}-"

        # Script-mode path: s3://{script_model_bucket}/model-outputs/{upload_directory}/{effective_date_suffix}/
        bucket_name = self.wallets_config['aws']['script_model_bucket']
        base_prefix = f"model-outputs/{self.upload_directory}/{effective_date_suffix}/"
        model_file_path = "output/model.tar.gz"

        # List all objects under the upload folder
        s3_client = self.sagemaker_session.boto_session.client('s3')

        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=base_prefix,
                Delimiter='/'
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise FileNotFoundError(f"Model bucket does not exist: {bucket_name}") from e
            else:
                raise ConfigError(f"Unable to access S3 bucket {bucket_name}: {e}") from e

        if 'CommonPrefixes' not in response:
            raise FileNotFoundError(f"No script-mode models found under path: s3://{bucket_name}/{base_prefix}")

        # Filter for training job folders matching our pattern
        matching_folders = []

        for prefix_info in response['CommonPrefixes']:
            folder_path = prefix_info['Prefix']
            folder_name = folder_path.rstrip('/').split('/')[-1]

            if folder_name.startswith(job_name_pattern):
                # Extract timestamp from folder name
                timestamp_part = folder_name[len(job_name_pattern):]
                matching_folders.append((timestamp_part, folder_name, folder_path))

        if not matching_folders:
            target_description = f"epoch_shift={epoch_shift}" if epoch_shift is not None else f"date_suffix='{self.date_suffix}'"
            raise FileNotFoundError(f"No script-mode models found for upload_directory '{self.upload_directory}' "
                                    f"and {target_description} "
                                    f"under path: s3://{bucket_name}/{base_prefix}")

        # Sort by timestamp to get most recent (assuming YYYYMMDD-HHMMSS format)
        matching_folders.sort(key=lambda x: x[0], reverse=True)
        most_recent_timestamp, most_recent_job_name, most_recent_folder_path = matching_folders[0]

        # Construct model URI and validate it exists
        model_uri = f"s3://{bucket_name}/{most_recent_folder_path}{model_file_path}"
        model_s3_key = f"{most_recent_folder_path}{model_file_path}"

        try:
            s3_client.head_object(Bucket=bucket_name, Key=model_s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError("Model file not found at expected location: "
                                        f"{model_uri}") from e
            else:
                raise ValueError(f"Unable to access model file at {model_uri}: {e}") from e

        # Store model artifacts
        self.model_uri = model_uri

        log_message = f"Loaded most recent script-mode model (timestamp: {most_recent_timestamp}): {model_uri}"
        logger.info(log_message)

        return {
            'model_uri': model_uri,
            'training_job_name': most_recent_job_name,
            'timestamp': most_recent_timestamp
        }


    def load_hpo_results(self, hpo_job_name: str = None) -> dict:
        """
        Load results from a completed HPO job.

        Params:
        - hpo_job_name (str, optional): Specific job name, or auto-detect most recent

        Returns:
        - dict: HPO results including best hyperparameters and training job details
        """
        if not hpo_job_name:
            hpo_job_name = self._find_most_recent_hpo_job()

        sagemaker_client = self.sagemaker_session.sagemaker_client

        # Get HPO job details
        hpo_response = sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=hpo_job_name
        )

        # Get best training job
        best_job_name = hpo_response['BestTrainingJob']['TrainingJobName']
        best_metrics = hpo_response['BestTrainingJob']['FinalHyperParameterTuningJobObjectiveMetric']

        # Get best job's hyperparameters
        training_response = sagemaker_client.describe_training_job(
            TrainingJobName=best_job_name
        )

        best_hyperparams = training_response['HyperParameters']
        model_uri = training_response['ModelArtifacts']['S3ModelArtifacts']

        return {
            'hpo_job_name': hpo_job_name,
            'best_training_job_name': best_job_name,
            'best_objective_value': best_metrics['Value'],
            'best_hyperparameters': best_hyperparams,
            'model_uri': model_uri,
            'hpo_status': hpo_response['HyperParameterTuningJobStatus'],
            'total_training_jobs': hpo_response['TrainingJobStatusCounters']['Completed']
        }

    def _find_most_recent_hpo_job(self) -> str:
        """Find the most recent HPO job for this upload_directory and date_suffix."""
        sagemaker_client = self.sagemaker_session.sagemaker_client

        # List HPO jobs with our naming pattern
        job_prefix = f"whpo-{self.upload_directory[:8]}"

        response = sagemaker_client.list_hyper_parameter_tuning_jobs(
            NameContains=job_prefix,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if not response['HyperParameterTuningJobSummaries']:
            raise FileNotFoundError(f"No HPO jobs found with prefix: {job_prefix}")

        return response['HyperParameterTuningJobSummaries'][0]['HyperParameterTuningJobName']


    @u.timing_decorator
    def predict_with_batch_transform(
            self,
            dataset_type: str = 'val',
            download_preds: bool = True,
            job_name_suffix: str = None,
            override_existing: bool = False
        ):
        """
        Score specified dataset using trained model via SageMaker batch transform.

        Params:
        - dataset_type (str): Type of dataset to score ('val' or 'test')
        - download_preds (bool): Whether to download the predictions to the local
            s3_downloads directory
        - job_name_suffix (str): Optional suffix to append to job name for uniqueness

        Returns:
        - dict: Contains transform job name and output S3 URI
        """
        # Validate URIs
        if not self.model_uri:
            raise ValueError("No trained model available. Call train_model() or "
                            "load_existing_model() first.")

        if not self.s3_uris:
            raise ConfigError("No S3 URIs available. Ensure training data has been configured.")

        date_uris = self.s3_uris[self.date_suffix]
        if dataset_type not in date_uris:
            raise FileNotFoundError(f"{dataset_type} data URI not found for date {self.date_suffix}")

        # Identify model name (i.e. the directory preceding '/output/model.tar.gz')
        if not self.model_uri.endswith('/output/model.tar.gz'):
            raise ValueError(f"Expected model URI to end with '/output/model.tar.gz', "
                            f"got: {self.model_uri}")
        model_name = self.model_uri.split('/')[-3]

        # Setup model for batch transform
        self._setup_model_for_batch_transform(model_name)

        # Execute batch transform on specified dataset
        dataset_uri = date_uris[dataset_type]
        result = self._execute_batch_transform(
            dataset_uri,
            model_name,
            override_existing=override_existing,
            job_name_suffix=job_name_suffix
        )

        # Download if configured
        if download_preds:
            self._download_batch_transform_preds(result['predictions_uri'], dataset_type)

        return result


    @u.timing_decorator
    def batch_predict_test_and_val(self, overwrite_existing: bool = False) -> dict[str, dict]:
        """
        Run batch transform predictions for 'test' and 'val' in parallel.
        Returns:
        - dict: Mapping split name ('test' or 'val') to the batch transform result dict.
        """
        splits = ['test', 'val']
        results: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_split = {
                executor.submit(
                    self.predict_with_batch_transform,
                    dataset_type=split,
                    download_preds=True,
                    job_name_suffix=f"concat-{split}",
                    override_existing=overwrite_existing
                ): split
                for split in splits
            }
            for future in as_completed(future_to_split):
                split = future_to_split[future]
                results[split] = future.result()

        return results


    def download_existing_model(self) -> str:
        """
        Download and extract model artifacts from S3 to persistent models directory.

        Returns:
        - str: Local path to extracted XGBoost model file

        Raises:
        - ValueError: If no model URI available
        - FileNotFoundError: If model artifacts not found at S3 location
        """
        if not self.model_uri:
            raise ValueError("No model URI available. Call train_model() or "
                             "load_existing_model() first.")

        # Parse S3 URI
        if not self.model_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {self.model_uri}")

        uri_parts = self.model_uri.replace('s3://', '').split('/', 1)
        bucket_name = uri_parts[0]
        s3_key = uri_parts[1]

        # Create models directory matching training data structure
        load_folder = self.wallets_config['training_data']['local_directory']

        models_dir = Path('../models') / load_folder
        models_dir.mkdir(parents=True, exist_ok=True)

        tar_path = models_dir / 'model.tar.gz'

        # Download if not already exists
        if not tar_path.exists():
            s3_client = boto3.client('s3')

            try:
                logger.info(f"Downloading model from {self.model_uri}")
                s3_client.download_file(bucket_name, s3_key, str(tar_path))
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise FileNotFoundError(f"Model not found at {self.model_uri}") from e
                else:
                    raise
        else:
            logger.info(f"Using existing model archive: {tar_path}")

        # Extract tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(models_dir)

        # Find the actual model file
        model_files = list(models_dir.glob('xgboost-model*'))
        if not model_files:
            model_files = list(models_dir.glob('*.model'))
        if not model_files:
            raise FileNotFoundError(f"No XGBoost model file found in {models_dir}")

        model_path = str(model_files[0])
        logger.info(f"Model ready at: {model_path}")

        return model_path



    def predict_using_endpoint(self, df: pd.DataFrame, df_type: str) -> np.ndarray:
        """
        Send a feature-only DataFrame to the deployed SageMaker endpoint for prediction.

        Params:
        - df (DataFrame): Preprocessed DataFrame with no target column, no headers, and
            correct column order.
        - df_type (str): Indicates which dataset the predictions were generated from
            (e.g., 'val' or 'test'). Used in the output filename as y_pred_{df_type}.

        Returns:
        - np.ndarray: Model predictions.
        Raises:
        - ValueError: If df_type is not 'val' or 'test'.
        """
        # Validate df_type
        if df_type not in ('val', 'test'):
            raise ValueError(f"Invalid df_type: {df_type}. Must be 'val' or 'test'.")
        # Null check
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        if df.isnull().values.any():
            raise ValueError("Input DataFrame contains null values, which are not allowed "
                             "for predictions with SageMaker.")

        if not self.endpoint_name:
            # Use helper to find existing endpoint if possible
            endpoint = self._find_existing_endpoint()
            if endpoint:
                self.endpoint_name = endpoint
                logger.info(f"Using detected endpoint: {self.endpoint_name}")
            else:
                raise ValueError("No endpoint deployed and none matched by prefix. "
                                 "Call deploy_endpoint() first.")

        # Estimate average row size
        sample_csv = df.head(100).to_csv(index=False, header=False).encode("utf-8")
        avg_row_size = max(1, len(sample_csv) // 100)

        # Set payload size limit - the hard cap is 6MB
        max_payload_bytes = 6 * 1024 * 1024
        # Set chunk size to this % of the hard cap
        buffer_ratio = 0.85
        rows_per_chunk = max(1, (max_payload_bytes * (buffer_ratio)) // avg_row_size)

        # Request confirmation based on total payload size and rows
        estimated_total_bytes = len(df) * avg_row_size
        estimated_total_mb = estimated_total_bytes / (1024 * 1024)
        total_chunks = max(1, len(df) // rows_per_chunk)
        logger.info(f"Prediction preview: {len(df)} rows across {total_chunks} chunks "
              f"({estimated_total_mb:.2f}MB estimated total size)")

        confirmation = u.request_confirmation(
            "Proceed with prediction?",
            approval_override=self.override_approvals
        )

        if confirmation.lower() != 'y':
            logger.info("Prediction cancelled by user")
            return np.array([])

        logger.info(f"Beginning endpoint predictions for {total_chunks} chunks...")
        u.notify('logo_corporate_warm_swell')
        predictor = Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=self.sagemaker_session,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer()
        )

        # Chunk and predict
        predictions = []
        for chunk in np.array_split(df, max(1, len(df) // rows_per_chunk)):
            payload = chunk.to_csv(index=False, header=False)
            preds = predictor.predict(payload)
            if isinstance(preds, dict) and 'predictions' in preds:
                predictions.extend([row['score'] for row in preds['predictions']])
            else:
                raise ValueError(f"Unexpected prediction format from endpoint: {preds[0]}")

        result_array = np.array(predictions)

        # Save predictions using helper
        self._save_endpoint_predictions(predictions, df_type)

        logger.info("Endpoint predictions completed successfully.")
        u.notify('mellow_chime_005')

        return result_array




    # -------------------------------
    #     Batch Transform Helpers
    # -------------------------------

    def _setup_model_for_batch_transform(self, model_name: str):
        """
        Create and register SageMaker model for batch transform jobs.

        Params:
        - model_name (str): the name of the trained model, which is passed through
            as the name of the scoring model instance
        """
        # Retrieve XGBoost container image
        xgb_container = sagemaker.image_uris.retrieve(
            framework=self.modeling_config['framework']['name'],
            region=self.sagemaker_session.boto_region_name,
            version=self.modeling_config['framework']['version']
        )

        # Create SageMaker model
        scoring_model = Model(
            image_uri=xgb_container,
            model_data=self.model_uri,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            name=model_name
        )

        # Register model in SageMaker
        scoring_model.create()


    def _execute_batch_transform(
            self,
            dataset_uri: str,
            model_name: str,
            override_existing: bool = False,
            job_name_suffix: str = None
        ):
        """
        Execute batch transform job for specified dataset URI.

        Params:
        - dataset_uri (str): S3 URI of dataset to score
        - model_name (str): Name of registered SageMaker model
        - override_existing (bool): Whether to overwrite existing output files
        - job_name_suffix (str): Optional suffix to append to job name for uniqueness

        Returns:
        - dict: Contains transform job name and output S3 URI
        """
        # Configure batch transform job
        timestamp = datetime.now().strftime("%H%M%S")
        job_name = f"wallet-scoring-{self.date_suffix}-{timestamp}"
        if job_name_suffix:
            job_name = f"{job_name}-{job_name_suffix}"

        output_path = (f"s3://{self.wallets_config['aws']['training_bucket']}/"
                    f"validation-data-scored/"
                    f"{self.upload_directory}/"
                    f"{self.date_suffix}/"
                    f"{job_name}")

        # Check if output already exists
        predictions_uri = f"{output_path}/{dataset_uri.split('/')[-1]}.out"

        if s3u.check_if_uri_exists(predictions_uri):
            # File exists
            if not override_existing:
                logger.info(f"Output already exists at: {predictions_uri}. Using existing file.")
                self.predictions_uri = predictions_uri
                return {
                    'transform_job_name': None,
                    'predictions_uri': predictions_uri,
                    'input_data_uri': dataset_uri,
                    'status': 'existing_file_used'
                }
            else:
                logger.warning(f"Output exists at: {predictions_uri}. Overwriting...")

        transformer = Transformer(
            model_name=model_name,
            instance_count=self.modeling_config['metaparams']['instance_count'],
            instance_type=self.modeling_config['metaparams']['batch_trans_instance_type'],
            output_path=output_path,
            sagemaker_session=self.sagemaker_session
        )

        # Execute batch transform
        logger.info(f"Starting batch transform job: {job_name}")
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Input data: {dataset_uri}")
        logger.debug(f"Output path: {output_path}")

        transformer.transform(
            data=dataset_uri,
            content_type='text/csv',
            split_type='Line',
            job_name=job_name,
            wait=True,
            logs=False,
            compression_type='Gzip',
            # Exclude the offset_days column (ID/filter col) from scoring input
            # For CSV, rows are treated as JSON arrays; $[1:] selects cols 1..end
            input_filter='$[1:]'
        )

        # Store predictions URI
        self.predictions_uri = predictions_uri

        logger.info(f"Batch transform completed. Predictions at: {predictions_uri}")

        result = {
            'transform_job_name': job_name,
            'predictions_uri': predictions_uri,
            'input_data_uri': dataset_uri
        }
        return result


    def _download_batch_transform_preds(self, predictions_uri: str, dataset_type: str) -> str:
        """
        Download batch transform predictions to standardized local path.

        Params:
        - predictions_uri (str): S3 URI of predictions file
        - dataset_type (str): Type of dataset ('val' or 'test')

        Returns:
        - str: Local file path where predictions were downloaded
        """
        # Construct standardized local path
        local_path = (f"{self.wallets_config['training_data']['local_s3_root']}/"
                      f"s3_downloads/wallet_predictions/"
                      f"{self.wallets_config['training_data']['download_directory']}/"
                      f"{self.date_suffix}/"
                      f"{dataset_type}.csv.out")

        # Use generic utility to download
        s3u.download_from_uri(predictions_uri, local_path)

        return local_path


    # -------------------------------
    #    Endpoint Utility Methods
    # -------------------------------

    def deploy_endpoint(self) -> str:
        """
        Deploy the trained model to a SageMaker real-time endpoint with a deterministic name
         and timestamp.

        Returns:
        - endpoint_name (str): The name of the deployed endpoint.
        """
        if not self.model_uri:
            raise ValueError("No model URI available. Call train_model() or "
                             "load_existing_model() first.")

        # Check for existing endpoint with matching prefix
        existing_endpoint = self._find_existing_endpoint()
        if existing_endpoint:
            logger.warning("An existing active endpoint matches the deployment prefix: "
                           f"{existing_endpoint}")
            confirmation = u.request_confirmation(
                f"Endpoint '{existing_endpoint}' already exists. "
                    "Deploy a new endpoint anyway?",
                approval_override=self.override_approvals
            )
            if confirmation.lower() != 'y':
                logger.info("Deployment cancelled by user; using existing endpoint.")
                self.endpoint_name = existing_endpoint
                return existing_endpoint

        # Retrieve the image URI for the model
        image_uri = sagemaker.image_uris.retrieve(
            framework=self.modeling_config['framework']['name'],
            region=self.sagemaker_session.boto_region_name,
            version=self.modeling_config['framework']['version']
        )

        # Generate endpoint name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"{self._get_endpoint_prefix()}-{timestamp}"

        # Check for active endpoints to avoid orphaned resources
        active_endpoints = self.list_active_endpoints()
        if len(active_endpoints) > 1:
            logger.warning(f"{len(active_endpoints)} active endpoints detected: {active_endpoints}")
        all_endpoints = self.list_all_endpoints()
        if len(all_endpoints) > 10:
            logger.warning(f"Found {all_endpoints} total endpoints. Consider cleaning "
                           "up old endpoints.")

        # Create the model object
        model = Model(
            image_uri=image_uri,
            model_data=self.model_uri,
            role=self.role,
            sagemaker_session=self.sagemaker_session
        )

        # Deploy the model to a real-time endpoint
        logger.info(f"Deploying real-time endpoint: {endpoint_name}...")
        predictor = model.deploy(
            initial_instance_count=self.modeling_config['metaparams']['instance_count'],
            instance_type=self.modeling_config['metaparams']['instance_type'],
            endpoint_name=endpoint_name
        )

        # Store state
        self.endpoint_name = endpoint_name
        self.predictor = predictor

        logger.info(f"Endpoint deployed: {self.endpoint_name}.")
        return self.endpoint_name


    def list_active_endpoints(self) -> list:
        """
        List currently active SageMaker endpoints.

        Returns:
        - list: Names of all active endpoints.
        """
        response = self.sagemaker_session.sagemaker_client.list_endpoints()
        active_endpoints = [ep['EndpointName'] for ep in response['Endpoints']]
        logger.info(f"Active endpoints: {active_endpoints}")
        return active_endpoints


    def list_all_endpoints(self) -> list:
        """
        List all SageMaker endpoints (active and inactive).

        Returns:
        - list: All endpoint names.
        """
        paginator = self.sagemaker_session.sagemaker_client.get_paginator('list_endpoints')
        all_endpoints = []
        for page in paginator.paginate():
            all_endpoints.extend(ep['EndpointName'] for ep in page['Endpoints'])
        return all_endpoints


    def delete_endpoint(self, endpoint_name: str):
        """
        Delete a specific SageMaker endpoint by name.

        Params:
        - endpoint_name (str): Name of the endpoint to delete.
        """
        try:
            logger.info(f"Deleting endpoint: {endpoint_name}")
            self.sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Successfully deleted endpoint: {endpoint_name}")
        except ClientError as e:
            logger.warning(f"Failed to delete endpoint {endpoint_name}: {e}")


    def delete_all_endpoints(self):
        """
        Delete all active SageMaker endpoints.
        """
        endpoints = self.list_active_endpoints()
        for endpoint_name in endpoints:
            self.delete_endpoint(endpoint_name)


    def _find_existing_endpoint(self) -> Optional[str]:
        """
        Search for an existing endpoint matching the expected prefix.
        Returns:
            str: endpoint name if exactly one match, None if no matches or multiple matches.
        """
        prefix = self._get_endpoint_prefix()
        candidates = self.list_active_endpoints()
        matching = [ep for ep in candidates if ep.startswith(prefix)]
        if len(matching) == 1:
            return matching[0]
        elif len(matching) > 1:
            logger.warning(f"Multiple active endpoints match prefix '{prefix}': {matching}")
            return None
        else:
            return None


    def _save_endpoint_predictions(self, predictions: list, df_type: str) -> None:
        """
        Save endpoint predictions to a CSV file in the configured endpoint_preds_dir.

        Params:
        - predictions (list): List of prediction scores.
        - df_type (str): Indicates which dataset the predictions were generated from
            (e.g., 'val' or 'test'). Used in the output filename as y_pred_{df_type}.
        """
        output_dir = Path(self.modeling_config["metaparams"]["endpoint_preds_dir"])
        if not output_dir.parent.exists():
            raise FileNotFoundError(f"Required directory '{output_dir.parent}/' not found.")
        output_dir.mkdir(parents=True, exist_ok=True)

        local_dir = self.wallets_config["training_data"]["local_directory"]
        output_file = (output_dir /
                       f"endpoint_y_pred_{df_type}_{local_dir}_{self.date_suffix}.csv")
        pd.DataFrame(predictions, columns=["score"]).to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")


    def _get_endpoint_prefix(self) -> str:
        """
        Generate deterministic endpoint name prefix based on framework and upload folder.
        """
        return f"{self.modeling_config['framework']['name']}-{self.upload_directory}"
