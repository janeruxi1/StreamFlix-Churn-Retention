"""Model training utilities for the StreamFlix churn pipeline.

Trainers + prep helper:
    prepare_features           -- raw df -> (X, y) model-ready matrix
    train_logistic_regression  -- regularized LR baseline
    train_xgboost              -- gradient-boosted trees
    train_random_forest        -- bagged trees, different bias-variance
    train_hist_gbm             -- sklearn's native GBM implementation
    tune_xgboost_optuna        -- Bayesian hyperparameter tuning
    calibrate_xgboost          -- Platt / isotonic post-hoc calibration

Design principles:
    - Stateless: no hidden train/test leakage. The one piece of state
      (one-hot column order) is returned as `feature_names` so the
      Streamlit app can align inference rows identically.
    - Calibration kept SEPARATE from the base model. We don't use
      `scale_pos_weight` -- it sacrifices calibration for ranking,
      and the Phase 6 decision rule needs calibrated probabilities.
    - LR and tree models share the same input matrix so the comparison
      is apples-to-apples.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------
DROP_COLS = [
    "subscriber_id", "monthly_revenue", "churned_next_30d",
    # Phase 8 uplift-experiment columns: not features, hold aside for
    # uplift training / evaluation only.
    "treated", "treatment_lever", "churned_if_treated",
    "y_observed", "true_uplift",
]
# monthly_revenue is collinear with plan_tier AND is the LTV input the
# Phase 6 decision rule consumes -- exclude from training features.

CATEGORICAL_COLS = [
    "plan_tier", "billing_cycle", "country", "payment_method",
    "engagement_cohort", "tenure_bucket",
]
BOOLEAN_COLS = [
    "auto_renew", "multi_profile", "promo_active",
    "is_trial_drop_window", "is_anniversary_window",
    "recent_plan_change_flag", "promo_expiring_soon_flag",
    "high_risk_segment_flag",
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert an engineered subscriber DataFrame into (X, y).

    - Drops IDs, target, and the LTV column (kept aside for decision rule)
    - One-hot encodes categoricals
    - Casts booleans to int
    - Preserves all other numerics as-is
    """
    y = df["churned_next_30d"].astype(int)

    feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for col in BOOLEAN_COLS:
        if col in feat.columns:
            feat[col] = feat[col].astype(int)

    feat = pd.get_dummies(feat, columns=CATEGORICAL_COLS, drop_first=False)

    for col in feat.columns:
        if feat[col].dtype == bool:
            feat[col] = feat[col].astype(int)

    return feat, y


# ---------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------
def train_logistic_regression(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              random_state: int = 42) -> Pipeline:
    """Regularized LR baseline with feature standardization.

    StandardScaler -> LogisticRegression with L2.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=2000,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  random_state: int = 42) -> XGBClassifier:
    """XGBoost classifier.

    No scale_pos_weight -- preserves calibration potential.
    Modest depth + many rounds + early shrinkage = standard tabular setup.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        random_state: int = 42) -> RandomForestClassifier:
    """Random Forest -- different bias-variance tradeoff vs boosted trees."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight=None,
    )
    model.fit(X_train, y_train)
    return model


def train_hist_gbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   random_state: int = 42) -> HistGradientBoostingClassifier:
    """Sklearn's HistGradientBoostingClassifier -- sklearn's native GBM,
    similar family to XGBoost but different implementation."""
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def tune_xgboost_optuna(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        n_trials: int = 25,
                        random_state: int = 42) -> Tuple[XGBClassifier, Dict]:
    """Bayesian hyperparameter tuning for XGBoost via Optuna.

    Optimizes PR-AUC on a held-out validation set. Optuna's TPE sampler
    focuses probe density in high-reward regions, so 25 trials get most
    of the value of a larger grid search at a fraction of the cost.

    Returns (fitted_best_model, best_params_dict).
    """
    try:
        import optuna
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise ImportError(
            "Optuna not installed. `pip install optuna` to enable "
            "hyperparameter tuning."
        )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, val_proba)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train, verbose=False)
    return best_model, study.best_params


# ---------------------------------------------------------------------
# Calibration wrapper
# ---------------------------------------------------------------------
def calibrate_xgboost(base_model: XGBClassifier,
                      X_calib: pd.DataFrame,
                      y_calib: pd.Series,
                      method: str = "sigmoid") -> CalibratedClassifierCV:
    """Wrap a prefit XGBoost with Platt (sigmoid) or isotonic calibration.

    Fits the calibrator on a held-out calibration set without retraining
    the base XGBoost. Compatible with both legacy sklearn (<1.6) using
    `cv='prefit'` and modern sklearn (>=1.6) using `FrozenEstimator`.

    Default 'sigmoid' (Platt): monotonic transform, preserves the model's
    ranking (so PR-AUC and ROC-AUC are unchanged), and smooths
    probabilities for the decision-rule's expected-value calc downstream.

    Isotonic is the alternative -- non-parametric and more flexible, but
    produces piecewise-constant probabilities which create ties and can
    measurably hurt PR-AUC on small positive classes.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit",
        )
    calibrated.fit(X_calib, y_calib)
    return calibrated
