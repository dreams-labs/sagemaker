# utilities/config_validation.py

import yaml

# Local module imports
from config_models.sage_wallets_config import SageWalletsConfig
from utils import ConfigError



# -------------------------
#     Primary Interface
# -------------------------


def load_sage_wallets_config(config_path: str) -> dict:
    """
    Load and validate SageMaker configuration from YAML file.

    Params:
    - config_path (str): Path to the sagemaker_config.yaml file

    Returns:
    - dict: Validated configuration dictionary

    Raises:
    - ConfigError: If file loading or validation fails
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            raw_config = yaml.safe_load(file)

        return validate_sage_wallets_config(raw_config)

    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML format: {str(e)}") from e
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {str(e)}") from e


def validate_sage_wallets_config(config: dict) -> dict:
    """
    Validate key parameters in the sage_wallets_config dictionary using pydantic.

    Params:
    - config (dict): Raw configuration dictionary from YAML

    Returns:
    - dict: Validated configuration dictionary

    Raises:
    - ConfigError: If any validation rule fails
    """
    try:
        # Apply pydantic validation
        _ = SageWalletsConfig(**config)

        # Return original dict after validation passes
        return config

    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {str(e)}") from e


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
