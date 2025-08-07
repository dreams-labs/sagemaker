"""
Validation logic for items in sagemaker_config.yaml
"""
from enum import Enum
from typing import Union, Dict, List
from pydantic import BaseModel, Field, field_validator


# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error messages
    }

# ============================================================================
#                   sagemaker_config.yaml Main Configuration
# ============================================================================

class SageWalletsConfig(NoExtrasBaseModel):
    """Top-level structure of the main sagemaker_config.yaml file."""
    training_data: 'TrainingDataConfig' = Field(...)
    preprocessing: 'PreprocessingConfig' = Field(...)
    workflow: 'WorkflowConfig' = Field(...)
    aws: 'AWSConfig' = Field(...)
    n_threads: 'NThreadsConfig' = Field(...)


# Training Data section
# ---------------------
class TrainingDataConfig(BaseModel):
    """
    Configuration for training data settings.
    """
    local_s3_root: str = Field(...)
    training_data_directory: str = Field(...)
    local_directory: str = Field(...)
    upload_directory: str = Field(...)
    dataset: str = Field(...)
    concatenate_offsets: bool = Field(...)
    date_0: int = Field(...)
    train_offsets: List[str] = Field(...)
    eval_offsets: List[str] = Field(...)
    test_offsets: List[str] = Field(...)
    val_offsets: List[str] = Field(...)

    @field_validator('local_directory')
    @classmethod
    def validate_local_directory(cls, v):
        """
        Validates that the local_directory string doesn't include hyphens.
        """
        if '-' in v:
            raise ValueError(f"Invalid local_directory value '{v}' contains hyphens. "
                              "Local folders should use underscores instead of hyphens.")

    @field_validator('upload_directory')
    @classmethod
    def validate_upload_directory(cls, v):
        """
        Validates that the upload_directory string doesn't exceed 20 characters. This
         param flows through as a filename component throughout the modeling process
         and a limit of 20 ensures that the SageMaker CreateTrainingJob operation
         doesn't fail because of a job name that exceeds the hard cap of 64 characters.
        """
        if '_' in v:
            raise ValueError(f"Invalid upload_directory value '{v}' contains underscores. "
                              "AWS syntax requires the use of hyphens instead of underscores.")

        if len(v) > 20:
            raise ValueError(f"'upload_directory' exceeds 20 characters (got {len(v)}): '{v}'")

        return v



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
                raise ValueError(f"fill_na value for '{key}' must be numeric or "
                                 f"string, got {type(value)}")
        return v


# AWS section
# -----------
class AWSConfig(NoExtrasBaseModel):
    """
    Configuration for AWS settings.
    """
    training_bucket: str = Field(...)
    script_model_bucket: str = Field(...)
    preprocessed_directory: str = Field(...)
    concatenated_directory: str = Field(...)
    temporal_cv_directory: str = Field(...)
    modeler_arn: str = Field(...)


# NThreads section
# ----------------
class NThreadsConfig(BaseModel):
    """
    Configuration for SageMaker threading settings.
    """
    upload_all_training_data: int = Field(...)
    train_all_models: int = Field(...)


# Workflow section
# ----------------
class WorkflowConfig(NoExtrasBaseModel):
    """
    Configuration for workflow settings.
    """
    override_existing_models: bool = Field(...,description="whether to train models "
                                           "for a date_suffix if one already exists")


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order
#  they were defined.
SageWalletsConfig.model_rebuild()
