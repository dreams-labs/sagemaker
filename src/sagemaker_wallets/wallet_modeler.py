"""
Class to manage all steps of the wallet model training and scoring.

This class handles data that has already been feature engineered into a df indexed on
a wallet-coin-offset_date tuple, with features already present as columns.

Interacts with:
---------------
WalletWorkflowOrchestrator: uses this class for model construction
"""
import logging
from typing import Dict
import pandas as pd

# Local modules
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletModeler:
    """
    Handles feature engineering, model training, and prediction generation
    for wallet-coin performance modeling.
    """
    def __init__(
            self,
            training_data: Dict[str, pd.DataFrame]
        ):
        self.training_data = training_data


    # ------------------------
    #      Public Methods
    # ------------------------

    def train_model(self):
        pass

    def generate_predictions(self):
        pass

    def evaluate_model(self):
        pass





    # ------------------------
    #      Helper Methods
    # ------------------------

    def _validate_training_data(self):
        """
        Validates training data structure and content.
        """
        required_keys = [
            'x_train', 'y_train', 'x_eval', 'y_eval',
            'x_test', 'y_test', 'x_val', 'y_val'
        ]

        # Check all required keys present
        missing_keys = [key for key in required_keys if key not in self.training_data]
        if missing_keys:
            raise ValueError(f"Missing required data splits: {missing_keys}")

        # Validate each DataFrame
        for key, df in self.training_data.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for {key}, got {type(df)}")
            if df.empty:
                raise ValueError(f"DataFrame {key} is empty")

        # Validate matching indices between X and y pairs
        splits = ['train', 'eval', 'test', 'val']
        for split in splits:
            x_key, y_key = f'x_{split}', f'y_{split}'
            x_df, y_df = self.training_data[x_key], self.training_data[y_key]

            if not x_df.index.equals(y_df.index):
                raise ValueError(f"Index mismatch between {x_key} and {y_key}")
            if len(x_df) != len(y_df):
                raise ValueError(f"Length mismatch between {x_key} and {y_key}")
