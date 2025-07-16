"""
Class to manage all steps of the wallet model training and scoring.

This class handles data that has already been feature engineered and uploaded to S3,
 indexed on a wallet-coin-offset_date tuple, with features already present as columns.

Interacts with:
---------------
WalletWorkflowOrchestrator: uses this class for model construction
"""
import logging
from typing import Dict
from datetime import datetime
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

# Local module imports
from utils import ConfigError
import sage_utils.config_validation as ucv


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
    """
    def __init__(
            self,
            wallets_config: Dict,
            modeling_config: Dict,
            s3_uris: Dict[str, Dict[str, str]],
            date_suffix: str
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

        # Store dataset and upload folder as instance state
        self.dataset = wallets_config['training_data'].get('dataset', 'dev')
        base_upload_folder = wallets_config['training_data']['upload_folder']
        if self.dataset == 'dev':
            base_upload_folder = f"{base_upload_folder}-dev"
        self.upload_folder = base_upload_folder

        # Model artifacts
        self.model_uri = None
        self.predictions_uri = None


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
        logger.info("Starting SageMaker training...")

        # Validate date suffix
        if self.date_suffix not in self.s3_uris:
            available_dates = list(self.s3_uris.keys())
            raise ConfigError(f"Date suffix '{self.date_suffix}' not found in S3 URIs. Available: {available_dates}")

        date_uris = self.s3_uris[self.date_suffix]

        # Validate required training data
        required_splits = ['train', 'eval']
        for split in required_splits:
            if split not in date_uris:
                raise ConfigError(f"{split.capitalize()} data URI not found for date {self.date_suffix}")

        # Configure estimator with basic hyperparameters
        model_container = sagemaker.image_uris.retrieve(
            framework=self.modeling_config['framework']['name'],
            region=self.sagemaker_session.boto_region_name,
            version=self.modeling_config['framework']['version']
        )

        # Create descriptive model output path
        model_output_path = (f"s3://{self.wallets_config['aws']['training_bucket']}/"
                             f"sagemaker-models/{self.upload_folder}/")

        # Check if model output path already exists
        s3_client = self.sagemaker_session.boto_session.client('s3')
        bucket_name = self.wallets_config['aws']['training_bucket']
        prefix = f"sagemaker-models/{self.upload_folder}/"

        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=1
            )

            if 'Contents' in response:
                confirmation = input(f"Model {model_output_path} already exists. "
                                     "Overwrite existing model? (y/N): ")
                if confirmation.lower() != 'y':
                    logger.info("Training cancelled by user")
                    return {}
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchBucket':
                logger.warning(f"Unable to check existing models: {e}")

        xgb_estimator = Estimator(
            image_uri=model_container,
            instance_type=self.modeling_config['training']['instance_type'],
            instance_count=self.modeling_config['training']['instance_count'],
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            hyperparameters=self.modeling_config['training']['hyperparameters'],
            output_path=model_output_path
        )

        # Define training data inputs
        train_input = TrainingInput(
            s3_data=date_uris['train'],
            content_type='text/csv'
        )

        validation_input = TrainingInput(
            s3_data=date_uris['eval'],
            content_type='text/csv'
        )

        # Launch training job with descriptive name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"wallet-xgb-{self.upload_folder}-{self.date_suffix}-{timestamp}"

        logger.info(f"Launching training job: {job_name}")
        logger.info(f"Model output path: {model_output_path}")

        xgb_estimator.fit(
            inputs={
                'train': train_input,
                'validation': validation_input
            },
            job_name=job_name,
            wait=True
        )

        # Store training artifacts
        self.model_uri = xgb_estimator.model_data

        logger.info(f"Training completed. Model stored at: {self.model_uri}")

        return {
            'model_uri': self.model_uri,
            'training_job_name': job_name,
            'date_suffix': self.date_suffix
        }


    def load_existing_model(self):
        """
        Load the most recent trained model for a given date_suffix.

        Returns:
        - dict: Contains model URI and training job name of most recent model
        """
        bucket_name = self.wallets_config['aws']['training_bucket']
        base_prefix = f"sagemaker-models/{self.upload_folder}/"

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
                raise FileNotFoundError(f"Training bucket does not exist: {bucket_name}") from e
            else:
                raise ConfigError(f"Unable to access S3 bucket {bucket_name}: {e}") from e

        if 'CommonPrefixes' not in response:
            raise FileNotFoundError(f"No models found under path: s3://{bucket_name}/{base_prefix}")

        # Filter for training job folders matching our pattern
        job_name_pattern = f"wallet-xgb-{self.upload_folder}-{self.date_suffix}-"
        matching_folders = []

        for prefix_info in response['CommonPrefixes']:
            folder_path = prefix_info['Prefix']
            folder_name = folder_path.rstrip('/').split('/')[-1]

            if folder_name.startswith(job_name_pattern):
                # Extract timestamp from folder name
                timestamp_part = folder_name[len(job_name_pattern):]
                matching_folders.append((timestamp_part, folder_name, folder_path))

        if not matching_folders:
            raise FileNotFoundError(f"No models found for upload_folder '{self.upload_folder}' "
                                    f"and date_suffix '{self.date_suffix}' "
                                    f"under path: s3://{bucket_name}/{base_prefix}")

        # Sort by timestamp to get most recent (assuming YYYYMMDD-HHMMSS format)
        matching_folders.sort(key=lambda x: x[0], reverse=True)
        most_recent_timestamp, most_recent_job_name, most_recent_folder_path = matching_folders[0]

        # Construct model URI and validate it exists
        model_uri = f"s3://{bucket_name}/{most_recent_folder_path}output/model.tar.gz"
        model_s3_key = f"{most_recent_folder_path}output/model.tar.gz"

        try:
            s3_client.head_object(Bucket=bucket_name, Key=model_s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(f"Model file not found at expected location: {model_uri}") from e
            else:
                raise ValueError(f"Unable to access model file at {model_uri}: {e}") from e

        # Store model artifacts
        self.model_uri = model_uri

        logger.info(f"Loaded most recent model (timestamp: {most_recent_timestamp}): {model_uri}")

        return {
            'model_uri': model_uri,
            'training_job_name': most_recent_job_name,
            'timestamp': most_recent_timestamp
        }


    def predict_with_cloud_model(self):
        """
        Score validation data using trained model via SageMaker batch transform.

        Returns:
        - dict: Contains transform job name and output S3 URI
        """
        if not self.model_uri:
            raise ValueError("No trained model available. Call train_model() or load_existing_model() first.")

        # Use date_suffix from instance variable
        if not self.s3_uris:
            raise ConfigError("No S3 URIs available. Ensure training data has been configured.")

        date_suffix = self.date_suffix
        date_uris = self.s3_uris[date_suffix]

        if 'val' not in date_uris:
            raise FileNotFoundError(f"Validation data URI not found for date {date_suffix}")

        # Create model for batch transform
        xgb_container = sagemaker.image_uris.retrieve(
            framework=self.modeling_config['framework']['name'],
            region=self.sagemaker_session.boto_region_name,
            version=self.modeling_config['framework']['version']
        )

        model_name = f"wallet-model-{self.upload_folder}"

        model = Model(
            image_uri=xgb_container,
            model_data=self.model_uri,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            name=model_name
        )

        # Register model in SageMaker
        model.create()

        # Configure batch transform job
        timestamp = datetime.now().strftime("%H%M%S")
        job_name = f"wallet-scoring-{date_suffix}-{timestamp}"

        output_path = f"s3://{self.wallets_config['aws']['training_bucket']}/validation-data-scored/"

        transformer = Transformer(
            model_name=model_name,
            instance_count=self.modeling_config['predicting']['instance_count'],
            instance_type=self.modeling_config['predicting']['instance_type'],
            output_path=output_path,
            sagemaker_session=self.sagemaker_session
        )

        # Deploy model if needed
        logger.info(f"Starting batch transform job: {job_name}")
        logger.info(f"Using model: {model_name}")
        logger.info(f"Input data: {date_uris['val']}")
        logger.info(f"Output path: {output_path}")

        # Start batch transform
        transformer.transform(
            data=date_uris['val'],
            content_type='text/csv',
            split_type='Line',
            job_name=job_name,
            wait=True
        )

        # Store predictions URI
        self.predictions_uri = f"{output_path}{job_name}/{date_uris['val'].split('/')[-1]}.out"

        logger.info(f"Batch transform completed. Predictions at: {self.predictions_uri}")

        return {
            'transform_job_name': job_name,
            'predictions_uri': self.predictions_uri,
            'input_data_uri': date_uris['val']
        }
