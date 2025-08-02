"""
Thin wrapper to launch SageMaker script-mode XGBoost training.
"""

import logging
from datetime import datetime
from typing import Dict, Union

from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#   Shared helper functions for script-mode XGBoost training
# ---------------------------------------------------------------------------
def _prepare_hyperparameters(raw_hp: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    """
    Remap and filter raw hyperparameters for script-mode.
    """
    hp: Dict[str, Union[int, float]] = {}
    # remap rounds
    if 'num_round' in raw_hp:
        hp['num_boost_round'] = raw_hp['num_round']
    elif 'num_boost_round' in raw_hp:
        hp['num_boost_round'] = raw_hp['num_boost_round']

    # allowed flags
    allowed = {'num_boost_round', 'eta', 'max_depth', 'subsample', 'early_stopping_rounds', 'score_threshold'}
    for key in allowed - {'num_boost_round'}:
        if key in raw_hp:
            hp[key] = raw_hp[key]

    # warn unsupported
    unsupported = set(raw_hp.keys()) - {'num_round', 'num_boost_round'} - allowed
    if unsupported:
        logger.warning(f"Ignoring unsupported hyperparameters: {unsupported}")
    return hp


def _build_job_name(prefix: str, upload_dir: str, suffix: str = None) -> str:
    """
    Build a unique SageMaker job name and return job_name.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [prefix, upload_dir]
    if suffix:
        parts.append(suffix)
    parts.append(ts)
    job_name = "-".join(parts)
    return job_name


def train_single_period_script_model(
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
    eval_uri = date_uris.get('eval')
    if not eval_uri:
        raise ValueError(f"'eval' data URI missing for date_suffix {date_suffix}")

    # Read script_mode configuration
    script_cfg = wallets_config.get('script_mode', {})
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
    job_name = _build_job_name("wscript", upload_dir, date_suffix)

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


# -----------------------------------------------------------
#   Multi-period temporal cross-validation script-mode trainer
# -----------------------------------------------------------
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
    script_cfg = wallets_config.get('script_mode', {})
    entry_point = script_cfg['entry_point']
    source_dir = script_cfg['source_dir']

    # Prepare hyperparameters for script-mode
    hp = _prepare_hyperparameters(modeling_config['training']['hyperparameters'])

    # Build output path and job name
    bucket = wallets_config['aws']['script_model_bucket']
    upload_dir = wallets_config['training_data']['upload_directory']
    job_name = _build_job_name("wallet-script-cv", upload_dir)
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
