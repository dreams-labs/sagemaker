"""
Orchestrates evaluation and analysis tasks for trained wallet models.

This class handles post-training analysis including feature importance extraction,
model performance evaluation, and cross-model comparison. It works with models
that have already been trained by WalletWorkflowOrchestrator.
"""
import sys
import logging
import concurrent.futures
import json
import tarfile
import tempfile
from pathlib import Path
import pandas as pd
import boto3
import xgboost as xgb

# Local module imports
from sage_wallet_modeling.wallet_modeler import WalletModeler
import utils as u
from utils import ConfigError
import sage_utils.config_validation as ucv

# Import from data-science repo   # pylint:disable=wrong-import-position
sys.path.append(str(Path("..") / ".." / "data-science" / "src"))
import wallet_insights.wallet_validation_analysis as wiva

# Set up logger at the module level
logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates evaluation and analysis of trained wallet models.

    Handles feature importance analysis, model comparison, and performance
    evaluation across multiple epoch shifts or model variants.

    Params:
    - wallets_config (dict): Configuration for model locations and data paths
    - modeling_config (dict): Configuration for model parameters and evaluation
    """

    def __init__(self, wallets_config: dict, modeling_config: dict):
        # Validate configs
        ucv.validate_sage_wallets_config(wallets_config)
        ucv.validate_sage_wallets_modeling_config(modeling_config)

        self.wallets_config = wallets_config
        self.modeling_config = modeling_config
        self.dataset = self.wallets_config['training_data'].get('dataset', 'dev')

        # Cache for loaded metadata and models
        self._metadata_cache = {}

        self._complete_importance_df = None


    # ------------------------
    #    Feature Importance Analysis
    # ------------------------

    @u.timing_decorator
    def generate_all_feature_importances(self) -> pd.DataFrame:
        """
        Extract and analyze feature importances across all epoch shift models.
        Generate the complete dataset once for multiple filtering operations.

        Returns:
        - DataFrame: Complete importance data for all models and features
        """
        epoch_shifts = self.wallets_config['training_data'].get('epoch_shifts', [])
        n_threads = self.wallets_config['n_threads'].get('analyze_all_importances', 4)

        if not epoch_shifts:
            raise ConfigError("No epoch_shifts configured in wallets_config")

        logger.milestone(f"Generating feature importances for {len(epoch_shifts)} epoch shifts...")

        # Extract importances for all models in parallel
        importance_results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_shift = {
                executor.submit(self._extract_and_analyze_single_importance, shift): shift
                for shift in epoch_shifts
            }

            for future in concurrent.futures.as_completed(future_to_shift):
                shift = future_to_shift[future]
                try:
                    result = future.result()
                    importance_results[shift] = result
                    logger.info(f"✓ Analyzed importances for epoch_shift={shift}")
                except Exception as e:
                    logger.error(f"✗ Failed importance analysis for epoch_shift={shift}: {e}")
                    importance_results[shift] = {'error': str(e)}

        # Filter out failed extractions
        successful_results = {k: v for k, v in importance_results.items() if 'error' not in v}

        if not successful_results:
            raise RuntimeError("No successful importance extractions found")

        logger.info(f"Successfully analyzed {len(successful_results)}/{len(epoch_shifts)} models")

        # Build complete dataset with all features and models
        all_feature_data = []

        for shift, result in successful_results.items():
            feature_importances_df = result['analyzed_importances']
            total_importance = result['total_importance']

            # Add model-level info to each feature row
            feature_df = feature_importances_df.copy()
            feature_df['epoch_shift'] = shift
            feature_df['total_model_importance'] = total_importance
            feature_df['sum_pct'] = feature_df['importance'] / total_importance

            all_feature_data.append(feature_df)

        # Combine all models into single DataFrame
        self._complete_importance_df = pd.concat(all_feature_data, ignore_index=True)

        logger.info(f"Generated complete importance dataset: "
                    f"{len(self._complete_importance_df)} feature-model combinations")

        return self._complete_importance_df


    def filter_and_aggregate_importances(
        self,
        feature_categories_filter: list = None,
        feature_names_filter: list = None,
        groups: list = None,
        complete_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Filter and aggregate the complete importance dataset.
        Aggregates to model-level totals first, then calculates distribution statistics.

        Params:
        - feature_categories_filter (list): Categories to include
        - feature_names_filter (list): Specific feature names to include
        - groups (list): Groupby columns for aggregation
        - complete_df (DataFrame): Pre-generated complete dataset, or None to use cached

        Returns:
        - DataFrame: Filtered and aggregated importance statistics across models
        """
        # Use provided df or cached version
        if complete_df is None:
            if not hasattr(self, '_complete_importance_df'):
                raise ValueError("No complete importance data available. "
                                 "Call generate_all_feature_importances() first.")
            complete_df = self._complete_importance_df

        # Default grouping
        if groups is None:
            groups = ['feature_category']

        # Apply filters
        filtered_df = complete_df.copy()

        if feature_categories_filter:
            filtered_df = filtered_df[filtered_df['feature_category'].isin(feature_categories_filter)]

        if feature_names_filter:
            filtered_df = filtered_df[filtered_df['feature_name'].isin(feature_names_filter)]

        # KEY CHANGE: First aggregate by model and group combination
        # This sums all features within each category for each model
        model_group_totals = (filtered_df
                            .groupby(['epoch_shift'] + groups)
                            .agg({
                                'importance': 'sum',
                                'total_model_importance': 'first'  # Same for all rows in a model
                            })
                            .reset_index())

        # Calculate percentage for each model-group combination
        model_group_totals['sum_pct'] = (model_group_totals['importance'] /
                                        model_group_totals['total_model_importance'])

        # Now calculate distribution statistics across models
        distribution_stats = (model_group_totals
                            .groupby(groups)['sum_pct']
                            .agg(['count', 'mean', 'median', 'min', 'max', 'std'])
                            .round(4))

        # Add total importance stats (across all models)
        total_importance_stats = (model_group_totals
                                .groupby(groups)['importance']
                                .agg(['sum', 'mean'])
                                .round(2))

        # Combine stats with better column ordering
        final_stats = pd.concat([distribution_stats, total_importance_stats], axis=1)
        final_stats.columns = [
            'model_count', 'pct_mean', 'pct_median', 'pct_min', 'pct_max', 'pct_std',
            'total_importance_sum', 'importance_mean'
        ]

        # Sort by mean percentage importance
        final_stats = final_stats.sort_values('pct_mean', ascending=False)

        logger.info(f"Filtered importance analysis: {len(final_stats)} feature "
                    f"groups across {model_group_totals['epoch_shift'].nunique()} models")

        return final_stats


    # Keep the old method for backward compatibility, but deprecate it
    @u.timing_decorator
    def analyze_all_feature_importances(
        self,
        feature_categories_filter: list = None,
        feature_names_filter: list = None,
        groups: list = None
    ) -> pd.DataFrame:
        """
        DEPRECATED: Use generate_all_feature_importances() + filter_and_aggregate_importances() instead.
        """
        logger.warning("analyze_all_feature_importances() is deprecated. Use generate + filter pattern for better performance.")
        complete_df = self.generate_all_feature_importances()
        return self.filter_and_aggregate_importances(
            feature_categories_filter, feature_names_filter, groups, complete_df
        )
    def extract_single_model_importances(self, model_uri: str) -> pd.DataFrame:
        """
        Extract feature importances from a single model URI.

        Params:
        - model_uri (str): S3 URI of the trained model

        Returns:
        - DataFrame: Feature importances with proper column names
        """
        return self._extract_feature_importances(model_uri)


    # ------------------------
    #    Helper Methods
    # ------------------------

    def _extract_and_analyze_single_importance(self, epoch_shift: int) -> dict:
        """Extract and analyze importances for a single epoch shift model."""
        # Load model info
        modeler = WalletModeler(
            wallets_config=self.wallets_config,
            modeling_config=self.modeling_config,
            date_suffix=f"sh{epoch_shift}",
            s3_uris=None,
            override_approvals=True
        )

        try:
            model_info = modeler.load_existing_model(epoch_shift=epoch_shift)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No model found for epoch_shift={epoch_shift}") from exc

        # Extract raw importances
        raw_importances = self._extract_feature_importances(model_info['model_uri'])

        # Analyze using existing function
        feature_importances_df = wiva.analyze_wallet_model_importance(raw_importances.to_dict('records'))

        return {
            'epoch_shift': epoch_shift,
            'model_uri': model_info['model_uri'],
            'analyzed_importances': feature_importances_df,
            'total_importance': feature_importances_df['importance'].sum()
        }


    def _aggregate_importance_distributions(
        self,
        importance_results: dict,
        feature_categories_filter: list,
        feature_names_filter: list,
        groups: list
    ) -> pd.DataFrame:
        """Aggregate importance distributions across all models."""

        all_aggregations = []

        # Process each model's importances
        for shift, result in importance_results.items():
            feature_importances_df = result['analyzed_importances']
            total_importance = result['total_importance']

            # Apply filters (same logic as your current approach)
            filtered_df = feature_importances_df.copy()

            if feature_categories_filter:
                filtered_df = filtered_df[filtered_df['feature_category'].isin(feature_categories_filter)]

            if feature_names_filter:
                filtered_df = filtered_df[filtered_df['feature_name'].isin(feature_names_filter)]

            # Group and aggregate
            grouped = (filtered_df
                    .fillna('None')
                    .groupby(groups)['importance']
                    .agg(['sum', 'count'])
                    .reset_index())

            # Calculate percentage of total model importance
            grouped['sum_pct'] = grouped['sum'] / total_importance
            grouped['epoch_shift'] = shift

            all_aggregations.append(grouped)

        # Combine all models
        combined_df = pd.concat(all_aggregations, ignore_index=True)

        # Calculate distribution statistics across models
        distribution_stats = (combined_df
                            .groupby(groups)['sum_pct']
                            .agg(['count', 'min', 'max', 'mean', 'median', 'std'])
                            .round(4))

        # Add total importance stats
        total_importance_stats = (combined_df
                                .groupby(groups)['sum']
                                .agg(['sum', 'mean'])
                                .round(2))

        # Combine stats
        final_stats = pd.concat([distribution_stats, total_importance_stats], axis=1)
        final_stats.columns = [
            'model_count', 'pct_min', 'pct_max', 'pct_mean', 'pct_median', 'pct_std',
            'total_importance_sum', 'importance_mean'
        ]

        # Sort by mean percentage importance
        final_stats = final_stats.sort_values('pct_mean', ascending=False)

        logger.info(f"Aggregated importance analysis complete: {len(final_stats)} feature groups analyzed")

        return final_stats


    def _extract_feature_importances(self, model_uri: str) -> pd.DataFrame:
        """
        Extract feature importances from SageMaker model URI.

        Returns:
        - DataFrame: Features and their importance scores, sorted by importance
        """
        logger.debug(f"Extracting feature importances from: {model_uri}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Parse S3 URI and download model
            bucket, key = model_uri.replace('s3://', '').split('/', 1)
            s3_client = boto3.client('s3')

            tar_path = Path(temp_dir) / 'model.tar.gz'
            s3_client.download_file(bucket, key, str(tar_path))

            # Extract model archive
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(temp_dir)

            # Find and load XGBoost model
            model_files = list(Path(temp_dir).glob('xgboost-model*'))
            if not model_files:
                raise FileNotFoundError(f"No XGBoost model found in {model_uri}")

            booster = xgb.Booster()
            booster.load_model(str(model_files[0]))

            # Extract importances
            importance_dict = booster.get_score(importance_type='gain')

            # Load metadata to map feature indices to names
            metadata = self._load_concatenated_metadata()
            feature_names = metadata['feature_columns'][1:]  # Skip offset_date column

            # Build importance DataFrame with proper feature names
            importance_data = []
            for i, feat_name in enumerate(feature_names):
                xgb_feat_key = f'f{i}'  # XGBoost uses f0, f1, f2...
                if xgb_feat_key in importance_dict:
                    importance_data.append({
                        'feature': feat_name,
                        'importance': importance_dict[xgb_feat_key],
                        'feature_index': i
                    })

            importances_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)

            logger.info(f"Extracted {len(importances_df)} feature importances for model")
            return importances_df


    def _load_concatenated_metadata(self) -> dict:
        """Load metadata from concatenated dataset, with caching."""
        if 'concatenated' not in self._metadata_cache:
            base_dir = Path(self.wallets_config['training_data']['local_s3_root'])
            concat_dir = base_dir / 's3_uploads' / 'wallet_training_data_concatenated'
            local_dir = self.wallets_config['training_data']['local_directory']
            metadata_path = concat_dir / local_dir / 'metadata.json'

            with open(metadata_path, 'r', encoding='utf-8') as f:
                self._metadata_cache['concatenated'] = json.load(f)

        return self._metadata_cache['concatenated']
