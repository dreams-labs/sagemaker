"""
Validation logic for items in sagemaker_modeling_config.yaml
"""
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",      # Prevent extra fields that are not defined
        "str_max_length": 2000, # Max length of error message strings
    }

# ============================================================================
#          sage_wallet_modeling_modeling_config.yaml Main Configuration
# ============================================================================

class SageWalletsModelingConfig(NoExtrasBaseModel):
    """Top-level structure of the main sagemaker_modeling_config.yaml file."""
    framework: 'FrameworkConfig' = Field(...)
    metaparams: 'MetaparamsConfig' = Field(...)
    script_mode: 'ScriptModeConfig' = Field(...)
    target: 'TargetConfig' = Field(...)
    training: 'TrainingConfig' = Field(...)
    predicting: 'PredictingConfig' = Field(...)


# [Framework]
# -----------
class FrameworkName(str, Enum):
    """Supported ML framework names."""
    XGBOOST = "xgboost"

class FrameworkVersion(str, Enum):
    """Supported framework versions."""
    XGBOOST_1_7_1 = "1.7-1"

class FrameworkConfig(BaseModel):
    """
    Configuration for ML framework settings.
    """
    name: FrameworkName = Field(...)
    version: FrameworkVersion = Field(...)


# [Metaparams]
# ------------
class MetaparamsConfig(BaseModel):
    """
    Configuration for metaparameters.
    """
    endpoint_preds_dir: str = Field(...)
    instance_type: str = Field(...)
    instance_count: int = Field(...)


    @field_validator('endpoint_preds_dir')
    @classmethod
    def validate_endpoint_preds_dir(cls, v):
        """
        Validates that the endpoint_preds_dir string doesn't include hyphens.
        """
        if '-' in v:
            raise ValueError(f"Invalid endpoint_preds_dir value '{v}' contains hyphens. "
                              "Local folders should use underscores instead of hyphens.")


# [Script_mode]
# -------------
class ScriptModeConfig(NoExtrasBaseModel):
    """
    Configuration for script-mode training settings.
    """
    enabled: bool = Field(..., description="Whether to enable script-mode training")
    entry_point: str = Field(..., description="Script entry point for training")
    source_dir: str = Field(..., description="Path to source directory for script-mode training")


# [Training]
# ----------
class ModelType(str, Enum):
    """Enum for self[training][model_type]."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

# Insert CustomValConfig before TargetConfig for readability and model building
class CustomValConfig(NoExtrasBaseModel):
    """
    Configuration for custom validation strategy.
    """
    metric: str = Field(..., description="Metric used to evaluate predictions during custom validation")
    top_quantile: float = Field(..., description="Percentage of samples to apply custom validation to")
    top_scores: float = Field(..., description="Minimum model score to consider for custom validation")

class TrainingConfig(NoExtrasBaseModel):
    """
    Configuration for training settings.
    """
    model_type: 'ModelType' = Field(
        ..., description="Type of model: 'regression' or 'classification'"
    )
    eval_metric: str = Field(...)
    hyperparameters: dict = Field(...)
    custom_val: CustomValConfig = Field(
        ..., description="Custom validation configuration"
    )
    custom_filters: Optional[dict] = Field(...)
    hpo: dict = Field(...)


# [Predicting]
# ------------
class PredictingConfig(NoExtrasBaseModel):
    """
    Configuration for prediction settings.
    """
    y_pred_threshold: float = Field(...)


# [Target]
# --------
class ClassificationConfig(NoExtrasBaseModel):
    """
    Configuration for self[target][classification]
    """
    threshold: float = Field(
        ..., description="Threshold for classification target: if target > threshold, class = 1"
    )


class TargetConfig(NoExtrasBaseModel):
    """
    Configuration for target settings.
    """
    classification: ClassificationConfig = Field(
        ..., description="Classification target settings"
    )
    target_var: str = Field(
        ..., description="Target variable used for training"
    )


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
SageWalletsModelingConfig.model_rebuild()
