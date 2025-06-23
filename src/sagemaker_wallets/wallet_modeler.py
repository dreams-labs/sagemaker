"""
Class to manage all steps of the wallet model training, including preprocessing,
modeling, and scoring.

This class handles data that has already been feature engineered into a df indexed on
a wallet-coin-offset_date tuple, with features already present as columns.

Interacts with:
---------------
WalletWorkflowOrchestrator: uses this class for model construction
"""
from pathlib import Path





class WalletModeler:
    """
    Handles feature engineering, model training, and prediction generation
    for wallet-coin performance modeling.
    """
    def __init__(
        self,
        data_folder: Path
    ):
        self.data_folder = data_folder
        self._validate_data_folder()




    # ------------------------
    #      Public Methods
    # ------------------------

    def preprocess_features(self):
        pass

    def train_model(self):
        pass

    def generate_predictions(self):
        pass

    def evaluate_model(self):
        pass





    # ------------------------
    #      Helper Methods
    # ------------------------

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
