"""
Orchestrates the full wallet modeling workflow data loading management,
model training coordination, and results handling.
"""
import logging
from pathlib import Path
import pandas as pd


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
    def __init__(self):

        # Training data variables
        self.training_data = None
        self.data_folder = None



    # ------------------------
    #      Public Methods
    # ------------------------

    def load_training_data(
            self,
            data_folder: Path,
            date_suffixes: list
        ):
        """
        Load and combine training data across multiple prediction period dates.

        Params:
        - data_folder (Path): Relative location of the training data parquet files
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
        self.data_folder = data_folder
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


    def run_training_pipeline(self):
        """
        Trains models for all configured scenarios.
        """
        modeler = WalletModeler()
        modeler.set_training_data(self.training_data)


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
