"""Uplift (causal) modeling for the StreamFlix retention pipeline.

Motivation:
    The Phase 4 model predicts P(churn | user). The Phase 6 decision rule
    multiplies that by an ASSUMED uplift constant per lever (e.g., 15%
    for credit_5). Two problems with that assumption:

      1. Real uplift is heterogeneous -- a payment-issue user responds
         very differently to a credit than a heavy-viewer does.
      2. Some users are "sleeping dogs": intervention makes them MORE
         likely to churn (the "we miss you" nudge reminds them to
         cancel). A propensity-only model can't spot these.

    Uplift modeling estimates the CAUSAL treatment effect per user
    (P(churn | treated) - P(churn | control)), from a randomized
    experiment. Feeding that into the decision rule instead of the
    constant lets us target true persuadables and skip sleeping dogs.

Trainers here (all wrap scikit-uplift):
    train_s_learner    -- Single-model approach (treatment as feature).
                           Cheapest, but can under-fit heterogeneity.
    train_t_learner    -- Two independent models, one on treated, one on
                           control. Uplift = P_control - P_treated.
    train_x_learner    -- Two-stage propensity-weighted variant. Robust
                           when treatment/control sizes are unbalanced.
    train_class_transformation -- Reformulates uplift as a single
                           classification problem via label transformation.

All trainers accept (X, treatment, y) and return a fitted model with a
`predict(X)` method that returns per-user uplift predictions in the
sklift convention: `P(Y=1 | T=1) - P(Y=1 | T=0)`.

    Since Y=1 means churn in this project, a "good" (churn-reducing)
    intervention produces NEGATIVE predicted uplift. When reporting to
    business stakeholders, flip the sign: reduction = -uplift.

Design principles (same as train.py):
    - Stateless. All state is inside the returned model object.
    - scikit-uplift is a HARD dependency (unlike shap/mlflow). The
      Phase 8 notebook needs it to run.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# scikit-uplift is a hard dep; fail fast with a clear message if missing
try:
    from sklift.models import SoloModel, TwoModels, ClassTransformation
except ImportError as e:
    raise ImportError(
        "scikit-uplift is required for src.models.uplift. "
        "Install with `pip install scikit-uplift`."
    ) from e


# ---------------------------------------------------------------------
# Default base estimator
# ---------------------------------------------------------------------
def _default_base_estimator(random_state: int = 42):
    """Base learner used inside every uplift meta-model.

    HistGradientBoosting is a good default: handles mixed dtypes, no
    scaling required, similar-family to XGBoost so the results are
    comparable to Phase 4, and it's part of sklearn so no extra
    dependency risk.
    """
    return HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=4,
        learning_rate=0.08,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------
def train_s_learner(X_train: pd.DataFrame,
                    treatment_train: np.ndarray,
                    y_train: np.ndarray,
                    random_state: int = 42) -> SoloModel:
    """S-learner: one model with treatment as a feature.

    Uplift = model.predict_proba(X, T=1) - model.predict_proba(X, T=0).

    Cheapest option -- one fit, one prediction. But if the base model
    doesn't split heavily on the treatment feature, it will underfit
    heterogeneous treatment effects.
    """
    model = SoloModel(estimator=_default_base_estimator(random_state))
    model.fit(X_train, y_train, treatment_train)
    return model


def train_t_learner(X_train: pd.DataFrame,
                    treatment_train: np.ndarray,
                    y_train: np.ndarray,
                    random_state: int = 42) -> TwoModels:
    """T-learner: two independent models, one per treatment arm.

    Fits P(Y|T=1, X) on treated users and P(Y|T=0, X) on control users.
    Uplift = P(Y|T=1, X) - P(Y|T=0, X).

    Simple and interpretable. Weakness: when one arm is much smaller
    than the other, the smaller-arm model is noisier -- but our 50/50
    split doesn't have that problem.
    """
    model = TwoModels(
        estimator_trmnt=_default_base_estimator(random_state),
        estimator_ctrl=_default_base_estimator(random_state),
        method="vanilla",
    )
    model.fit(X_train, y_train, treatment_train)
    return model


def train_x_learner(X_train: pd.DataFrame,
                    treatment_train: np.ndarray,
                    y_train: np.ndarray,
                    random_state: int = 42) -> TwoModels:
    """X-learner (Kunzel et al. 2019): two-stage variant that shares
    information between the treatment arms.

    Uses `ddr_control` (Difference-in-Differences with control response)
    from sklift, which uses the control-arm model to impute the
    counterfactual for treated users, then fits a second-stage model to
    the residuals. More robust than T-learner when the CATE surface is
    smoother than the outcome surfaces themselves.
    """
    model = TwoModels(
        estimator_trmnt=_default_base_estimator(random_state),
        estimator_ctrl=_default_base_estimator(random_state),
        method="ddr_control",
    )
    model.fit(X_train, y_train, treatment_train)
    return model


def train_class_transformation(X_train: pd.DataFrame,
                               treatment_train: np.ndarray,
                               y_train: np.ndarray,
                               random_state: int = 42) -> ClassTransformation:
    """Class-transformation approach (Jaskowski & Jaroszewicz 2012).

    Creates a new binary target Z such that P(Z=1) - 0.5 is proportional
    to the treatment effect, then fits a standard classifier. Requires a
    50/50 treatment split (which our simulator produces).

    Elegant: reduces uplift modeling to a single standard classification
    problem. Good baseline to compare against T-/X-learners.
    """
    model = ClassTransformation(
        estimator=_default_base_estimator(random_state),
    )
    model.fit(X_train, y_train, treatment_train)
    return model


# ---------------------------------------------------------------------
# Convenience: score all users with a fitted uplift model
# ---------------------------------------------------------------------
def predict_uplift(model, X: pd.DataFrame) -> np.ndarray:
    """Return per-user uplift predictions.

    Convention (sklift): uplift = P(Y=1 | T=1) - P(Y=1 | T=0).
    In this project Y=1 means churn, so a useful retention treatment
    produces NEGATIVE uplift. To convert to a "reduction" score
    (higher = better), negate.
    """
    return model.predict(X)
