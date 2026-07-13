"""SHAP explainability utilities for the StreamFlix churn model.

SHAP (SHapley Additive exPlanations) attributes a specific prediction
back to feature contributions, in a mathematically principled way based
on cooperative game theory. Each feature gets a signed contribution --
positive pushes the prediction toward CHURN, negative pushes toward RETAIN.

For tree models (XGBoost) we use `TreeExplainer` which is exact and fast.
Values are in the model's LOG-ODDS output space; sum(shap_values, axis=1)
+ expected_value == raw model output (before the calibration wrapper).

This module deliberately doesn't touch the sklearn CalibratedClassifierCV
wrapper. We compute SHAP on the underlying uncalibrated XGBoost, which
tells the correct explanatory story (which features drive risk) even
though final ranked probabilities come from the calibrated model.

Functions:
    compute_shap_values(model, X)          -- returns shap.Explanation
    global_importance(shap_values, ...)    -- ranked mean |SHAP| table
    local_explanation(shap_values, X, ...) -- top contributors for one row
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

# shap is imported lazily inside compute_shap_values() so that the
# rest of this module (FEATURE_INTERVENTION_MAP, map_to_intervention,
# global_importance, local_explanation) is usable in environments
# without the heavy shap/numba dependency chain -- notably CI.
if TYPE_CHECKING:
    import shap  # noqa: F401


def compute_shap_values(model, X: pd.DataFrame,
                        sample_size: Optional[int] = None,
                        random_state: int = 42):
    """Run TreeExplainer on an XGBoost model.

    For a large test set we can subsample to speed things up -- the
    explanations don't need every row. Set sample_size=None to use
    every row.
    """
    import shap  # local import: heavy dep, not needed for the intervention map
    if sample_size is not None and sample_size < len(X):
        X = X.sample(n=sample_size, random_state=random_state)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


def global_importance(shap_values,
                      feature_names: List[str],
                      top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean |SHAP value| across the sample.

    This is the CORRECT way to measure feature importance for decisions
    -- unlike XGBoost's built-in `feature_importances_`, which reports
    training-time split gain (biased toward high-cardinality features).
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    mean_signed = shap_values.values.mean(axis=0)
    out = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    })
    out = out.sort_values("mean_abs_shap", ascending=False).head(top_n)
    out = out.reset_index(drop=True)
    return out


def local_explanation(shap_values,
                      X: pd.DataFrame,
                      feature_names: List[str],
                      idx: int,
                      top_n: int = 8) -> pd.DataFrame:
    """Top-N feature contributions for a single row.

    Returns a DataFrame with:
        feature       -- name of the feature
        feature_value -- the actual value for this user
        shap_value    -- signed contribution to log-odds (positive = churn)
        direction     -- 'RISK+' or 'RISK-' for quick reading
    """
    row_shap = shap_values.values[idx]
    row_vals = X.iloc[idx].values if hasattr(X, "iloc") else X[idx]

    out = pd.DataFrame({
        "feature": feature_names,
        "feature_value": row_vals,
        "shap_value": row_shap,
    })
    out["abs_shap"] = out["shap_value"].abs()
    out = out.sort_values("abs_shap", ascending=False).head(top_n)
    out["direction"] = np.where(out["shap_value"] > 0, "RISK+", "RISK-")
    out = out.drop(columns=["abs_shap"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Feature -> intervention mapping
# ---------------------------------------------------------------------
# Curated by the Retention PM (Marcus Lee). Each entry maps a top
# feature back to a concrete lever the team can actually pull. Features
# NOT in this table are considered non-actionable (informational only)
# so the model exposes them but the decision rule doesn't act on them.
FEATURE_INTERVENTION_MAP = {
    "watch_trend_7d_to_30d": {
        "lever": "Personalized content push",
        "cost": 1.0,
        "note": "Declining engagement -> curated playlist push notification",
    },
    "watch_trend_30d_to_90d": {
        "lever": "Personalized content push",
        "cost": 1.0,
        "note": "Slow-burn disengagement -> re-engagement email + trailer bundle",
    },
    "days_since_last_login": {
        "lever": "Re-engagement email sequence",
        "cost": 1.0,
        "note": "Inactive users need a reason to come back",
    },
    "logins_last_30d": {
        "lever": "Re-engagement email sequence",
        "cost": 1.0,
        "note": "Low login frequency -> weekly recap emails",
    },
    "support_tickets_7d": {
        "lever": "White-glove support callback",
        "cost": 5.0,
        "note": "Recent ticket escalation -> proactive support outreach",
    },
    "tickets_recency_ratio": {
        "lever": "White-glove support callback",
        "cost": 5.0,
        "note": "Sudden burst of tickets = frustration; catch before they cancel",
    },
    "payment_failures_30d": {
        "lever": "Payment method update prompt + $5 credit",
        "cost": 5.0,
        "note": "Failed payments often not a churn decision -- fix the bill first",
    },
    "payment_failures_recency_ratio": {
        "lever": "Payment method update prompt + $5 credit",
        "cost": 5.0,
        "note": "Recent failures spike -> emergency retry + card update flow",
    },
    "payment_health_score": {
        "lever": "Payment method update prompt + $5 credit",
        "cost": 5.0,
        "note": "Composite payment risk score",
    },
    "days_until_promo_expires": {
        "lever": "Extend promo or offer tier upgrade",
        "cost": 5.0,
        "note": "Users on the cusp of losing discounts -- prevent sticker shock",
    },
    "promo_expiring_soon_flag": {
        "lever": "Extend promo or offer tier upgrade",
        "cost": 5.0,
        "note": "Promo about to expire -> proactive renewal offer",
    },
    "promo_expiry_risk_score": {
        "lever": "Extend promo or offer tier upgrade",
        "cost": 5.0,
        "note": "Continuous promo-expiry risk signal",
    },
    "days_since_plan_change": {
        "lever": "Post-downgrade satisfaction check-in",
        "cost": 1.0,
        "note": "Recent plan change -> validate the switch was right",
    },
    "recent_plan_change_flag": {
        "lever": "Post-downgrade satisfaction check-in",
        "cost": 1.0,
        "note": "Same lever, boolean version",
    },
    "plan_change_risk_score": {
        "lever": "Post-downgrade satisfaction check-in",
        "cost": 1.0,
        "note": "Continuous plan-change risk signal",
    },
    "is_trial_drop_window": {
        "lever": "Trial-to-paid onboarding sequence",
        "cost": 1.0,
        "note": "Month 2 spike -- structured onboarding content",
    },
    "high_risk_segment_flag": {
        "lever": "Trial-to-paid onboarding + $5 credit",
        "cost": 5.0,
        "note": "The m2-casual cohort: 15% baseline churn, worth stacking interventions",
    },
    "auto_renew": {
        "lever": "Renewal confirmation dialog",
        "cost": 0.0,
        "note": "Hard to intervene on -- treat as diagnostic, not actionable",
    },
    "tenure_months": {
        "lever": "N/A -- structural signal",
        "cost": 0.0,
        "note": "Tenure is diagnostic, not an intervention target",
    },
    "billing_cycle": {
        "lever": "Upgrade-to-annual offer",
        "cost": 12.0,
        "note": "Convert monthly users to annual for higher retention",
    },
}


def map_to_intervention(feature_name: str) -> dict:
    """Look up the intervention lever for a feature.

    Returns a dict with 'lever', 'cost', 'note'. If no mapping exists
    (feature is structural/diagnostic), returns a 'no lever' placeholder.
    """
    # Strip one-hot suffix, e.g. 'plan_tier_Basic' -> 'plan_tier'
    key = feature_name.rsplit("_", 1)[0] if "_" in feature_name else feature_name
    for candidate in (feature_name, key):
        if candidate in FEATURE_INTERVENTION_MAP:
            return FEATURE_INTERVENTION_MAP[candidate]
    return {
        "lever": "N/A (diagnostic feature)",
        "cost": 0.0,
        "note": "Non-actionable signal -- model uses it for prediction only",
    }
