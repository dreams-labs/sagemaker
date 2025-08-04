"""
Thin wrapper to launch SageMaker script-mode XGBoost training.
"""

import logging
from datetime import datetime
from typing import Dict, Union

from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import IntegerParameter, ContinuousParameter, CategoricalParameter

# Local module imports
from script_modeling.entry_helpers import HYPERPARAMETER_TYPES
from utils import ConfigError

logger = logging.getLogger(__name__)



# ------------------------------------
#       Primary Interface Router
# ------------------------------------

def initiate_script_modeling(
    wallets_config: Dict,
    modeling_config: Dict,
    date_suffix: str,
    s3_uris: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    """
    Route to HPO or standard training based on modeling_config.training.hpo.enabled.

    Params:
    - wallets_config (dict): Validated sage_wallets_config.yaml
    - modeling_config (dict): Validated sage_wallets_modeling_config.yaml
    - date_suffix (str): Date string like 'YYMMDD'
    - s3_uris (dict): URIs for train/eval splits

    Returns:
    - dict: Training results (same format regardless of HPO vs standard)
    """
    hpo_config = modeling_config['training']['hpo']
    hpo_enabled = hpo_config.get('enabled', False)

    if hpo_enabled:
        logger.info(f"HPO enabled for {date_suffix}, launching hyperparameter optimization...")
        return _launch_hyperparameter_optimization(
            wallets_config, modeling_config, date_suffix, s3_uris
        )
    else:
        logger.info(f"HPO disabled for {date_suffix}, using standard training...")
        return _train_single_period_script_model(
            wallets_config, modeling_config, date_suffix, s3_uris
        )




# -------------------------------
#       Primary Helpers
# -------------------------------

def _train_single_period_script_model(
    wallets_config: Dict,
    modeling_config: Dict,
    date_suffix: str,
    s3_uris: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    """
    Launch a SageMaker script-mode XGBoost training job.

    Params:
    - wallets_config (dict): Validated sage_wallets_config.yaml as a dict.
    - modeling_config (dict): Validated sage_wallets_modeling_config.yaml as a dict.
    - date_suffix (str): Date string like 'YYMMDD' indicating the training slice.
    - s3_uris (dict): Mapping from date_suffix to URIs for splits:
        {'train': ..., 'val' or 'eval': ..., ...}

    Returns:
    - dict: {
        'model_uri': str,            # S3 URI of the trained model artifact
        'training_job_name': str,    # Name of the SageMaker training job
        'date_suffix': str           # Echoed date_suffix
      }
    """
    # Validate date_suffix presence
    if date_suffix not in s3_uris:
        raise ValueError(f"Date suffix {date_suffix} not found in s3_uris")
    date_uris = s3_uris[date_suffix]

    # Extract URIs for train and eval channels
    train_uri = date_uris['train']
    eval_uri = date_uris['eval']
    if not eval_uri:
        raise ValueError(f"'eval' data URI missing for date_suffix {date_suffix}")

    # Read script_mode configuration
    script_cfg = modeling_config['script_mode']
    entry_point = script_cfg['entry_point']
    source_dir = script_cfg['source_dir']

    # Build output path
    bucket = wallets_config['aws']['script_model_bucket']
    upload_dir = wallets_config['training_data']['upload_directory']
    output_path = f"s3://{bucket}/model-outputs/{upload_dir}/{date_suffix}"

    # Prepare hyperparameters for script-mode
    hp = _prepare_hyperparameters(modeling_config['training']['hyperparameters'])

    # Instantiate the XGBoost ScriptMode estimator
    estimator = XGBoost(
        entry_point=entry_point,
        source_dir=source_dir,
        framework_version=modeling_config['framework']['version'],
        instance_type=modeling_config['metaparams']['instance_type'],
        instance_count=modeling_config['metaparams']['instance_count'],
        role=wallets_config['aws']['modeler_arn'],
        hyperparameters=hp,
        output_path=output_path
    )

    # Assemble job name
    job_name = _build_job_name("wscr", upload_dir, date_suffix)

    # Prepare TrainingInput channels
    train_input = TrainingInput(s3_data=train_uri, content_type='text/csv')
    validation_input = TrainingInput(s3_data=eval_uri, content_type='text/csv')

    logger.info(f"Launching script-mode training job: {job_name}")
    estimator.fit(
        {"train": train_input, "validation": validation_input},
        job_name=job_name,
        wait=True
    )

    model_uri = estimator.model_data
    logger.info(f"Script-mode training completed. Model URI: {model_uri}")

    return {
        "model_uri": model_uri,
        "training_job_name": job_name,
        "date_suffix": date_suffix
    }


def _launch_hyperparameter_optimization(
    wallets_config: Dict,
    modeling_config: Dict,
    date_suffix: str,
    s3_uris: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    """
    Launch SageMaker hyperparameter optimization using config-driven parameters.

    Params:
    - wallets_config (dict): Validated sage_wallets_config.yaml
    - modeling_config (dict): Validated sage_wallets_modeling_config.yaml
    - date_suffix (str): Date string like 'YYMMDD'
    - s3_uris (dict): URIs for train/eval splits

    Returns:
    - dict: HPO job details and best training job info (same format as standard training)
    """
    # Extract HPO configuration
    hpo_config = modeling_config['training']['hpo']
    max_jobs = hpo_config['max_jobs']
    max_parallel_jobs = hpo_config['max_parallel_jobs']
    objective_metric_name = hpo_config['objective_metric_name']
    # Validate inputs using existing pattern
    if date_suffix not in s3_uris:
        raise ValueError(f"Date suffix {date_suffix} not found in s3_uris")
    date_uris = s3_uris[date_suffix]

    train_uri = date_uris['train']
    eval_uri = date_uris['eval']
    if not eval_uri:
        raise ValueError(f"'eval' data URI missing for date_suffix {date_suffix}")

    # Create base estimator using existing logic
    script_cfg = modeling_config['script_mode']
    bucket = wallets_config['aws']['script_model_bucket']
    upload_dir = wallets_config['training_data']['upload_directory']
    output_path = f"s3://{bucket}/hpo-outputs/{upload_dir}/{date_suffix}"

    # Define hyperparameter ranges from config
    hyperparameter_ranges = _get_hpo_parameter_ranges(modeling_config)

    # Create base estimator without fixed hyperparameters
    base_estimator = XGBoost(
        entry_point=script_cfg['entry_point'],
        source_dir=script_cfg['source_dir'],
        framework_version=modeling_config['framework']['version'],
        instance_type=modeling_config['metaparams']['instance_type'],
        instance_count=modeling_config['metaparams']['instance_count'],
        role=wallets_config['aws']['modeler_arn'],
        output_path=output_path
    )

    # Configure HPO tuner
    metric_definitions = [
        {
            'Name': 'validation:aucpr',  # Use colon for HPO
            'Regex': r'eval_aucpr:([0-9\.]+)'  # Extract from your print statement
        }
    ]

    tuner = HyperparameterTuner(
        estimator=base_estimator,
        objective_metric_name=objective_metric_name,
        objective_type='Maximize',  # We want to maximize PR-AUC
        metric_definitions=metric_definitions,
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs
    )

    # Build job name and prepare inputs
    job_name = _build_job_name("whpo", upload_dir, date_suffix)
    train_input = TrainingInput(s3_data=train_uri, content_type='text/csv')
    validation_input = TrainingInput(s3_data=eval_uri, content_type='text/csv')

    logger.info(f"Launching HPO job: {job_name}")
    logger.info(f"Max jobs: {max_jobs}, Max parallel: {max_parallel_jobs}")
    logger.info(f"Optimizing: {objective_metric_name}")

    # Launch HPO
    tuner.fit(
        {"train": train_input, "validation": validation_input},
        job_name=job_name,
        wait=True
    )

    # Get best training job results
    best_training_job = tuner.best_training_job()
    best_model_uri = f"{output_path}/{best_training_job}/output/model.tar.gz"

    logger.info(f"HPO completed. Best training job: {best_training_job}")
    logger.info(f"Best model URI: {best_model_uri}")

    return {
        "hpo_job_name": job_name,
        "best_training_job_name": best_training_job,
        "model_uri": best_model_uri,
        "date_suffix": date_suffix,
        "tuner": tuner  # Return tuner object for analysis
    }



def train_temporal_cv_script_model(
    wallets_config: Dict,
    modeling_config: Dict,
    cv_s3_uri: str,
) -> Dict[str, str]:
    """
    Launch a SageMaker script-mode XGBoost training job for cross-date CV.

    Params:
    - wallets_config (dict): Validated sage_wallets_config.yaml as a dict.
    - modeling_config (dict): Validated sage_wallets_modeling_config.yaml as a dict.
    - cv_s3_uri (str): S3 URI of the root CV directory (contains fold_{suffix}/train.csv and /validation.csv).

    Returns:
    - dict: {
        'model_uri': str,            # S3 URI of the trained model artifact
        'training_job_name': str      # Name of the SageMaker training job
      }
    """
    # Read script_mode configuration
    script_cfg = modeling_config['script_mode']
    entry_point = script_cfg['entry_point']
    source_dir = script_cfg['source_dir']

    # Prepare hyperparameters for script-mode
    hp = _prepare_hyperparameters(modeling_config['training']['hyperparameters'])

    # Build output path and job name
    bucket = wallets_config['aws']['script_model_bucket']
    upload_dir = wallets_config['training_data']['upload_directory']
    job_name = _build_job_name("wscv", upload_dir)
    output_path = f"s3://{bucket}/model-outputs/{upload_dir}/cv/{job_name}"

    # Instantiate the XGBoost ScriptMode estimator
    estimator = XGBoost(
        entry_point=entry_point,
        source_dir=source_dir,
        framework_version=modeling_config['framework']['version'],
        instance_type=modeling_config['metaparams']['instance_type'],
        instance_count=modeling_config['metaparams']['instance_count'],
        role=wallets_config['aws']['modeler_arn'],
        hyperparameters=hp,
        output_path=output_path
    )

    # Prepare CV channel input
    cv_input = TrainingInput(s3_data=cv_s3_uri, content_type='text/csv')

    logger.info(f"Launching script-mode CV training job: {job_name}")
    estimator.fit({'cv': cv_input}, job_name=job_name, wait=True)

    model_uri = estimator.model_data
    logger.info(f"Script-mode CV training completed. Model URI: {model_uri}")

    return {
        'model_uri': model_uri,
        'training_job_name': job_name
    }


# ---------------------------------------------------------------------------
#   Shared helper functions for script-mode XGBoost training
# ---------------------------------------------------------------------------
def _prepare_hyperparameters(raw_hp: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    """
    Remap num_round to num_boost_round for script-mode compatibility. Why?
     * Built-in container: Uses num_round
     * Script-mode: Uses num_boost_round
    """
    hp = raw_hp.copy()

    # Handle built-in vs script-mode naming difference
    if 'num_round' in hp and 'num_boost_round' not in hp:
        hp['num_boost_round'] = hp.pop('num_round')
    elif 'num_round' in hp:
        # Both present - remove num_round, keep num_boost_round
        hp.pop('num_round')

    return hp


def _build_job_name(prefix: str, upload_dir: str, suffix: str = None) -> str:
    """
    Build a unique SageMaker job name within 32 character limit.
    Reserves 11 chars for timestamp, leaving 21 for prefix-uploaddir-suffix.
    """
    ts = datetime.now().strftime("%m%d-%H%M%S")  # 11 chars
    max_non_timestamp = 21  # 32 - 11 = 21

    # Build the non-timestamp part
    parts = [prefix, upload_dir[:8]]
    if suffix:
        parts.append(suffix)

    non_timestamp_part = "-".join(parts)

    # Truncate if too long
    if len(non_timestamp_part) > max_non_timestamp:
        # Calculate how much to trim from upload_dir
        excess = len(non_timestamp_part) - max_non_timestamp
        upload_dir_truncated = upload_dir[:max(1, 8 - excess)]
        parts[1] = upload_dir_truncated
        non_timestamp_part = "-".join(parts)

    job_name = f"{non_timestamp_part}-{ts}"
    return job_name


def _get_hpo_parameter_ranges(modeling_config: Dict) -> Dict:
    """
    Extract hyperparameter ranges from modeling config.

    Expected config format:
    hpo:
      param_ranges:
        eta: [0.05, 0.2]                    # ContinuousParameter
        max_depth: [3, 8]                   # IntegerParameter
        booster: ["gbtree", "dart"]         # CategoricalParameter

    Returns:
    - dict: SageMaker parameter objects for HPO
    """
    hpo_config = modeling_config['training']['hpo']
    param_ranges_config = hpo_config['param_ranges']

    ranges = {}
    for param_name, range_values in param_ranges_config.items():

        # Get the type from centralized source
        if param_name not in HYPERPARAMETER_TYPES:
            raise ConfigError(f"Unknown hyperparameter '{param_name}'. Must be defined in HYPERPARAMETER_TYPES.")

        param_type = HYPERPARAMETER_TYPES[param_name]

        # Convert Python type to SageMaker parameter object
        if param_type == int:
            if len(range_values) != 2:
                raise ConfigError(f"Integer parameter '{param_name}' must be [min, max], got {range_values}")
            min_val, max_val = range_values
            ranges[param_name] = IntegerParameter(min_val, max_val)

        elif param_type == float:
            if len(range_values) != 2:
                raise ConfigError(f"Float parameter '{param_name}' must be [min, max], got {range_values}")
            min_val, max_val = range_values
            ranges[param_name] = ContinuousParameter(min_val, max_val)

        else:
            # Handle categorical if needed later
            raise ConfigError(f"Unsupported parameter type {param_type} for '{param_name}'")

    logger.info(f"HPO will tune: {list(ranges.keys())}")
    return ranges
