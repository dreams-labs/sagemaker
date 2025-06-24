"""
Class to manage all steps of the wallet model training and scoring.

This class handles data that has already been feature engineered into a df indexed on
a wallet-coin-offset_date tuple, with features already present as columns.

Interacts with:
---------------
WalletWorkflowOrchestrator: uses this class for model construction
"""



class WalletModeler:
    """
    Handles feature engineering, model training, and prediction generation
    for wallet-coin performance modeling.
    """
    def __init__(
        self,
    ):
        self.training_data = None
