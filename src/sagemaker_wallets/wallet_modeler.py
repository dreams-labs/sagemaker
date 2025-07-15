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


# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletModeler:
    """
    Handles model training and prediction generation for wallet-coin performance modeling
    using SageMaker XGBoost with S3 data sources.
    """
    def __init__(
            self,
            sage_wallets_config: Dict,
            s3_uris: Dict[str, Dict[str, str]]
        ):
        # Config
        self.sage_wallets_config = sage_wallets_config
        self.s3_uris = s3_uris

        # SageMaker setup
        self.sagemaker_session = sagemaker.Session()
        self.role = sage_wallets_config['aws']['modeler_arn']

        # Model artifacts
        self.model_uri = None
        self.training_job_name = None


    # ------------------------
    #      Public Methods
    # ------------------------

    def train_model(self, date_suffix: str):
        """
        Train XGBoost model using SageMaker's built-in XGBoost algorithm.
        Uses train/test splits with eval for early stopping.

        Params:
        - date_suffix (str): Specific date to train model for (required).

        Returns:
        - dict: Contains model URI and training job name
        """
        logger.info("Starting SageMaker XGBoost training")

        # Validate date suffix
        if date_suffix not in self.s3_uris:
            available_dates = list(self.s3_uris.keys())
            raise ConfigError(f"Date suffix '{date_suffix}' not found in S3 URIs. Available: {available_dates}")

        date_uris = self.s3_uris[date_suffix]

        # Validate required training data
        required_splits = ['train', 'eval']
        for split in required_splits:
            if split not in date_uris:
                raise ConfigError(f"{split.capitalize()} data URI not found for date {date_suffix}")

        # Configure XGBoost estimator with basic hyperparameters
        xgb_container = sagemaker.image_uris.retrieve(
            framework='xgboost',
            region=self.sagemaker_session.boto_region_name,
            version='1.5-1'
        )

        # Extract upload_folder from config for naming
        upload_folder = self.sage_wallets_config['training_data']['upload_folder']
        dataset = self.sage_wallets_config['training_data'].get('dataset', 'prod')

        if dataset == 'dev':
            upload_folder = f"{upload_folder}_dev"

        # Create descriptive model output path
        model_output_path = (f"s3://{self.sage_wallets_config['aws']['training_bucket']}/"
                             f"sagemaker-models/{upload_folder}/")

        # Check if model output path already exists
        s3_client = self.sagemaker_session.boto_session.client('s3')
        bucket_name = self.sage_wallets_config['aws']['training_bucket']
        prefix = f"sagemaker-models/{upload_folder}/"

        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=1
            )

            if 'Contents' in response:
                confirmation = input(f"Model {model_output_path} already exists. Overwrite existing model? (y/N): ")
                if confirmation.lower() != 'y':
                    logger.info("Training cancelled by user")
                    return {}
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchBucket':
                logger.warning(f"Unable to check existing models: {e}")

        xgb_estimator = Estimator(
            image_uri=xgb_container,
            instance_type='ml.m5.large',
            instance_count=1,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            hyperparameters={
                'objective': 'reg:squarederror',
                'num_round': 100,
                'max_depth': 6,
                'eta': 0.3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 10
            },
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
        job_name = f"wallet-xgb-{upload_folder}-{date_suffix}-{timestamp}"

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
        self.training_job_name = job_name

        logger.info(f"Training completed. Model stored at: {self.model_uri}")

        return {
            'model_uri': self.model_uri,
            'training_job_name': self.training_job_name,
            'date_suffix': date_suffix
        }


    def generate_predictions(self):
        """
        Generate predictions on validation set for future period performance assessment.
        """
        if not self.model_uri:
            raise ValueError("No trained model available. Call train_model() first.")

        # Implementation for batch transform or real-time inference
        pass


    def evaluate_model(self):
        """
        Evaluate model performance on validation set (future period data).
        """
        if not self.model_uri:
            raise ValueError("No trained model available. Call train_model() first.")

        # Implementation for validation metrics
        pass
