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
        "protected_namespaces": ()  # Ignores protected spaces for params starting with  "model_"
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

class FrameworkConfig(NoExtrasBaseModel):
    """
    Configuration for ML framework settings.
    """
    pass  # Placeholder for future validation


# Metaparams section
# ------------------

class MetaparamsConfig(NoExtrasBaseModel):
    """
    Configuration for metaparameters.
    """
    pass  # Placeholder for future validation


# Training section
# ----------------

class TrainingConfig(NoExtrasBaseModel):
    """
    Configuration for training settings.
    """
    pass  # Placeholder for future validation


# Predicting section
# ------------------

class PredictingConfig(NoExtrasBaseModel):
    """
    Configuration for prediction settings.
    """
    pass  # Placeholder for future validation


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
SageWalletsModelingConfig.model_rebuild()
