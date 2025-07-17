"""
Validation logic for items in sagemaker_modeling_config.yaml
"""
from enum import Enum
from typing import Optional, List, Dict
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representations
    }

# ============================================================================
#          sagemaker_wallets_modeling_config.yaml Main Configuration
# ============================================================================

class SageWalletsModelingConfig(NoExtrasBaseModel):
    """Top-level structure of the main sagemaker_modeling_config.yaml file."""
    framework: 'FrameworkConfig' = Field(...)
    metaparams: 'MetaparamsConfig' = Field(...)
    training: 'TrainingConfig' = Field(...)
    predicting: 'PredictingConfig' = Field(...)


# Framework section
# -----------------

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


# Metaparams section
# ------------------

class MetaparamsConfig(BaseModel):
    """
    Configuration for metaparameters.
    """
    pass  # Placeholder for future validation


# Training section
# ----------------

class TrainingConfig(BaseModel):
    """
    Configuration for training settings.
    """
    pass  # Placeholder for future validation


# Predicting section
# ------------------

class PredictingConfig(BaseModel):
    """
    Configuration for prediction settings.
    """
    pass  # Placeholder for future validation


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
SageWalletsModelingConfig.model_rebuild()
