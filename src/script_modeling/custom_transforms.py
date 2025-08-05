

import pandas as pd
import numpy as np
import xgboost as xgb



# ------------------------------------------------------------------------ #
#                    Helpers for Custom Transformations                                 #
# ------------------------------------------------------------------------ #

def preprocess_custom_labels(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Extract and preprocess target column from y_full data using modeling_config."""
    # Pull settings from config
    target_var = config["target"]["target_var"]
    threshold = config["target"]["classification"]["threshold"]
    model_type = config["training"]["model_type"]

    # Ensure the target column exists
    if target_var not in df.columns:
        raise KeyError(f"Expected target column '{target_var}' not found in DataFrame")
    target_series = df[target_var]

    # Apply classification threshold if configured
    if model_type == 'classification':
        processed_target = (target_series > threshold).astype(int)
    else:
        processed_target = target_series.astype(float)

    # Check for NaNs or infinite values in labels
    if pd.isnull(processed_target).any():
        raise ValueError("NaNs detected in processed label array")
    if not np.isfinite(processed_target.values).all():
        raise ValueError("Infinite values detected in processed label array")
    return processed_target.values


def merge_xy_dmatrix(df_x: pd.DataFrame, df_y: pd.Series, config: dict) -> xgb.DMatrix:
    """
    Build an XGBoost DMatrix by taking raw feature DataFrame and raw label Series,
    applying any custom-transform logic, and returning the DMatrix.
    """
    # Ensure feature DataFrame is not empty
    if df_x.shape[0] == 0:
        raise ValueError("Feature DataFrame is empty")

    # Check for NaNs in feature DataFrame
    if df_x.isnull().values.any():
        raise ValueError("NaNs detected in feature DataFrame")

    labels = preprocess_custom_labels(df_y, config)

    # Ensure labels array is not empty
    if labels.shape[0] == 0:
        raise ValueError("Processed label array is empty")

    # Ensure features and labels have matching lengths
    if df_x.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature rows ({df_x.shape[0]}) and label length ({labels.shape[0]}) must match."
        )

    return xgb.DMatrix(df_x.values, label=labels)
