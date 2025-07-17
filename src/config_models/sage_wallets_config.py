"""
Validation logic for items in sagemaker_config.yaml
"""
from enum import Enum
from typing import Union, Dict
from pydantic import BaseModel, Field, field_validator


# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representation
    }

# ============================================================================
#                   sagemaker_config.yaml Main Configuration
# ============================================================================

class SageWalletsConfig(NoExtrasBaseModel):
    """Top-level structure of the main sagemaker_config.yaml file."""
    training_data: 'TrainingDataConfig' = Field(...)
    preprocessing: 'PreprocessingConfig' = Field(...)
    aws: 'AWSConfig' = Field(...)


# Training Data section
# ---------------------

class TrainingDataConfig(BaseModel):
    """
    Configuration for training data settings.
    """
    local_directory: str = Field(...)
    upload_directory: str = Field(...)




# Preprocessing section
# ---------------------

class FillNaMethod(str, Enum):
    """Valid fill_na method values."""
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"

class PreprocessingConfig(NoExtrasBaseModel):
    """
    Configuration for preprocessing steps.
    """
    fill_na: Dict[str, Union[float, int, FillNaMethod]] = Field(...)

    @field_validator('fill_na')
    @classmethod
    def validate_fill_na_values(cls, v):
        """Validate that all fill_na values are either numeric or valid method strings."""
        for key, value in v.items():
            if isinstance(value, (int, float)):
                continue
            elif isinstance(value, str):
                if value not in ["min", "max", "mean", "median"]:
                    raise ValueError(f"Invalid fill_na method for '{key}': '{value}'. "
                                   "Must be numeric or one of: min, max, mean, median")
            else:
                raise ValueError(f"fill_na value for '{key}' must be numeric or string, got {type(value)}")
        return v


# AWS section
# -----------

class AWSConfig(NoExtrasBaseModel):
    """
    Configuration for AWS settings.
    """
    training_bucket: str = Field(...)
    preprocessed_directory: str = Field(...)
    modeler_arn: str = Field(...)


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
SageWalletsConfig.model_rebuild()
