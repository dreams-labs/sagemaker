import copy
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb


# ------------------------------------------------------------------------ #
#                             y Transformations                            #
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



# ------------------------------------------------------------------------ #
#                             X Transformations                            #
# ------------------------------------------------------------------------ #

def apply_cli_filter_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Apply CLI hyperparameter overrides to custom feature filter configuration.

    Params:
    - config (dict): Original modeling configuration
    - args (Namespace): Parsed hyperparameters from SageMaker

    Returns:
    - dict: Config with filter values overridden by hyperparameters
    """
    config_copy = copy.deepcopy(config)

    # Skip if custom_x not enabled
    if not config_copy.get('training', {}).get('custom_x', False):
        return config_copy

    custom_filters = config_copy['training'].get('custom_filters', {})
    if not custom_filters:
        return config_copy

    # Apply overrides based on cli fields in filter config
    overrides_applied = []
    for filter_col, filter_config in custom_filters.items():
        cli_name = filter_config.get('cli')
        if not cli_name:
            continue  # Skip filters without cli field

        # Check for min override: filter_{cli_name}_min
        min_param = f"filter_{cli_name}_min"
        if hasattr(args, min_param):
            override_value = getattr(args, min_param)
            config_copy['training']['custom_filters'][filter_col]['min'] = override_value
            overrides_applied.append(f"{min_param}={override_value}")

        # Check for max override: filter_{cli_name}_max
        max_param = f"filter_{cli_name}_max"
        if hasattr(args, max_param):
            override_value = getattr(args, max_param)
            config_copy['training']['custom_filters'][filter_col]['max'] = override_value
            overrides_applied.append(f"{max_param}={override_value}")

    if overrides_applied:
        print(f"Applied CLI filter overrides: {', '.join(overrides_applied)}")

    return config_copy


def apply_custom_feature_filters(df_x: pd.DataFrame, metadata: dict, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply custom row-level filters to feature DataFrame based on config rules.

    Params:
    - df_x (DataFrame): Raw feature data without headers
    - metadata (dict): Contains feature_columns list for column name mapping
    - config (dict): modeling_config with preprocessing.custom_x filter definitions

    Returns:
    - tuple: (filtered_df, row_mask) where row_mask indicates kept rows for Y alignment
    """
    # Get filter definitions
    custom_filters = config['training']['custom_filters']

    # Early escape if no filters configured
    if not custom_filters:  # Handles None, {}, and other falsy values
        print("No custom filters configured, returning unfiltered data")
        # Return all rows (no filtering applied)
        return df_x, np.ones(len(df_x), dtype=bool)

    feature_columns = metadata['feature_columns']

    # Validate feature list length matches X df
    if len(feature_columns) != df_x.shape[1]:
        raise ValueError(f"Feature column count mismatch: metadata has {len(feature_columns)} "
                        f"columns, DataFrame has {df_x.shape[1]} columns")

    # Assign column names from metadata
    df_x_named = df_x.copy()
    df_x_named.columns = feature_columns

    # Verify all filter keys exist in feature list
    missing_columns = []
    for filter_col in custom_filters.keys():
        if filter_col not in feature_columns:
            missing_columns.append(filter_col)

    if missing_columns:
        raise ValueError(f"Filter columns not found in features: {missing_columns}")

    # Initialize filter mask for this column (start with all True)
    combined_mask = np.ones(len(df_x_named), dtype=bool)
    filter_mask = np.ones(len(df_x_named), dtype=bool)
    initial_row_count = len(df_x_named)

    print(f"Starting with {initial_row_count:,} rows before custom filtering")

    # Apply each filter and track impact
    for filter_col, filter_rules in custom_filters.items():
        col_data = df_x_named[filter_col]

        # Validate numeric data
        if not pd.api.types.is_numeric_dtype(col_data):
            raise ValueError(f"Filter column '{filter_col}' contains non-numeric data")

        # Print column stats
        print(f"Column stats for {filter_col}:")
        print(f"  Min: {col_data.min():,.0f}")
        print(f"  Max: {col_data.max():,.0f}")

        # Apply min filter
        if 'min' in filter_rules:
            min_val = filter_rules['min']
            min_mask = col_data >= min_val
            filter_mask &= min_mask
            rows_removed_min = (~min_mask).sum()
            print(f"Filter '{filter_col}' min {min_val}: removed {rows_removed_min:,} rows")

        # Apply max filter
        if 'max' in filter_rules:
            max_val = filter_rules['max']
            max_mask = col_data <= max_val
            filter_mask &= max_mask
            rows_removed_max = (~max_mask).sum()
            print(f"Filter '{filter_col}' max {max_val}: removed {rows_removed_max:,} rows")

        # Combine with overall mask
        rows_before = combined_mask.sum()
        combined_mask &= filter_mask
        rows_after = combined_mask.sum()
        rows_removed_this_filter = rows_before - rows_after

        print(f"Filter '{filter_col}' total impact: removed {rows_removed_this_filter:,} rows "
              f"({rows_after:,} remaining)")

    # Final validation
    final_row_count = combined_mask.sum()
    total_removed = initial_row_count - final_row_count
    removal_pct = (total_removed / initial_row_count) * 100

    print(f"Custom filtering complete: {total_removed:,} rows removed ({removal_pct:.1f}%), "
          f"{final_row_count:,} rows remaining")

    if final_row_count == 0:
        raise ValueError("All rows filtered out by custom filters - filters may be too restrictive")

    # Apply mask and return
    filtered_df = df_x_named[combined_mask].reset_index(drop=True)

    return filtered_df, combined_mask




# ------------------------------------------------------------------------ #
#                              Main Interface                              #
# ------------------------------------------------------------------------ #

def build_custom_dmatrix(
        df_x: pd.DataFrame,
        df_y: pd.Series,
        config: dict,
        metadata: dict
    ) -> xgb.DMatrix:
    """
    Build XGBoost DMatrix from raw features and labels, applying custom transformations.

    Params:
    - df_x (DataFrame): Raw feature data without headers
    - df_y (Series): Raw label data
    - config (dict): modeling_config with custom transformation settings
    - metadata (dict): Contains feature_columns and preprocessing metadata

    Returns:
    - DMatrix: Ready for XGBoost training with aligned features and labels
    """
    # Ensure feature DataFrame is not empty
    if df_x.shape[0] == 0:
        raise ValueError("Feature DataFrame is empty")

    # Check for NaNs in feature DataFrame
    if df_x.isnull().values.any():
        raise ValueError("NaNs detected in feature DataFrame")

    # Apply custom X filtering if enabled
    if config.get('training', {}).get('custom_x', False):
        df_x_final, row_mask = apply_custom_feature_filters(df_x, metadata, config)
        df_y_final = df_y[row_mask].reset_index(drop=True)
    else:
        df_x_final = df_x
        df_y_final = df_y

    # Apply y transformation
    labels = preprocess_custom_labels(df_y_final, config)

    # Ensure labels array is not empty
    if labels.shape[0] == 0:
        raise ValueError("Processed label array is empty")

    # Ensure features and labels have matching lengths
    if df_x_final.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature rows ({df_x_final.shape[0]}) and label length ({labels.shape[0]}) must match."
        )

    return xgb.DMatrix(df_x_final.values, label=labels)
