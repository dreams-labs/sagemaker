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
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


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

    def train_model(self):
        """
        Train XGBoost model using SageMaker's built-in XGBoost algorithm.
        Uses train/test splits with eval for early stopping.
        """
        logger.info("Starting SageMaker XGBoost training")

        # Use first date suffix for MVP (can iterate over multiple dates later)
        date_suffix = list(self.s3_uris.keys())[0]
        date_uris = self.s3_uris[date_suffix]

        # Configure XGBoost estimator with basic hyperparameters
        xgb_container = sagemaker.image_uris.retrieve(
            framework='xgboost',
            region=self.sagemaker_session.boto_region_name,
            version='1.5-1'
        )

        xgb_estimator = Estimator(
            image_uri=xgb_container,
            instance_type='ml.m5.large',  # You have 15 available quota for this!
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
            output_path=f"s3://{self.sage_wallets_config['aws']['training_bucket']}/sagemaker-models/"
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

        # Launch training job with unique name
        timestamp = datetime.now().strftime("%H%M%S")
        job_name = f"wallet-xgb-{date_suffix}-{timestamp}"
        logger.info(f"Launching training job: {job_name}")

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
            'training_job_name': self.training_job_name
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
