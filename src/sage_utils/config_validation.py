# utilities/config_validation.py

# Local module imports
from utils import ConfigError



# -------------------------
#     Primary Interface
# -------------------------

def validate_sage_wallets_config(config: dict) -> None:
    """
    Validate key parameters in the sage_wallets_config dictionary.
    Raises:
        ValueError if any validation rule fails.
    """
    upload_folder = config.get("training_data", {}).get("upload_folder", "")
    _validate_upload_folder_name(upload_folder)


def validate_sage_wallets_modeling_config(modeling_config: dict) -> None:
    """
    Validate modeling configuration parameters.

    Params:
    - modeling_config (Dict): Modeling configuration to validate

    Raises:
    - ValueError: If configuration is invalid
    """
    # Validate framework
    if 'framework' not in modeling_config:
        raise ConfigError("Missing required 'framework' section in modeling config")

    framework = modeling_config['framework']
    if 'name' not in framework:
        raise ConfigError("Missing 'name' in framework configuration")

    if framework['name'] != 'xgboost':
        raise ConfigError(f"Unsupported framework: {framework['name']}. "
                         "Only 'xgboost' is currently supported")



# ------------------------
#     Helper Functions
# ------------------------

def _validate_upload_folder_name(
        upload_folder: str,
        max_len: int = 20
    ) -> None:
    """
    Validates that the upload_folder string doesn't exceed 25 characters. This
     param flows through as a filename component throughout the modeling process
     and a limit of 25 ensures that the SageMaker CreateTrainingJob operation
     doesn't fail because of a job name that exceeds the hard cap of 64 characters.
    """
    if '_' in upload_folder:
        raise ConfigError(f"Invalid upload_folder value '{upload_folder} contains underscores. "
                          "AWS syntax requires the use of hyphens instead of underscores.")

    if len(upload_folder) > max_len:
        raise ConfigError(
            f"'upload_folder' exceeds {max_len} characters (got {len(upload_folder)}): '{upload_folder}'"
        )
