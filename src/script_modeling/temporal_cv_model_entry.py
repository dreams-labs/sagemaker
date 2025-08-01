"""
Script-mode entry point for cross-date temporal CV training.
"""

import os
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import average_precision_score
import entry_helpers as h


def main() -> None:
    """
    Loop over folds in /opt/ml/input/data/cv, train and evaluate each,
    compute mean PR-AUC, and save the best model.
    """
    # Determine CV fold directories under the CV channel mount
    cv_root = Path("/opt/ml/input/data/cv")
    fold_dirs = sorted([d for d in cv_root.iterdir() if d.is_dir() and d.name.startswith("fold_")])

    # Parse training hyperparameters injected via SageMaker CLI
    args = h.load_hyperparams()

    # Initialize tracking of best model and fold scores
    best_score = -float("inf")
    best_model = None
    scores = []

    for fold_dir in fold_dirs:

        # Load CSVs into XGBoost DMatrix (first column is label)
        train_csv = fold_dir / "train.csv"
        val_csv   = fold_dir / "validation.csv"
        dm_train = h.load_csv_as_dmatrix(train_csv)
        dm_val   = h.load_csv_as_dmatrix(val_csv)

        # Build XGBoost parameters focusing on PR-AUC
        booster_params = h.build_booster_params(args)

        # Train the model with early stopping on the in-period hold-out
        booster = xgb.train(
            params=booster_params,
            dtrain=dm_train,
            num_boost_round=args.num_boost_round,
            evals=[(dm_val, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False,
        )

        # Compute PR-AUC on this fold’s validation set
        preds = booster.predict(dm_val)
        actuals = dm_val.get_label()
        pr_auc = average_precision_score(actuals, preds)

        # Record this fold’s PR-AUC
        scores.append(pr_auc)

        print(f"{fold_dir.name}:cv_auc_pr={pr_auc:.6f}")

        # Update best model if this fold achieved higher PR-AUC
        if pr_auc > best_score:
            best_score = pr_auc
            best_model = booster

    # Save final metrics and best model artifact for deployment
    model_output_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    h.print_metrics_and_save(best_model, scores, model_output_dir)


if __name__ == "__main__":
    main()
