"""Feature engineering transforms for the StreamFlix churn model.

Pure, idempotent functions that take a raw subscriber DataFrame and return
the same frame with engineered features added. Each feature group is its
own function so they can be unit-tested independently in Phase 8.

The feature groups (in evaluation order):
    1. engagement_features      -- trend ratio + intensity
    2. tenure_features          -- bucket + spike-window flags
    3. recency_features         -- ticket / payment recency ratios
    4. lifecycle_features       -- plan-change & promo risk windows
    5. composite_features       -- high-risk segment + payment-health score

Design principles:
    - Pure: never mutate the input DataFrame; always return a new one.
    - Idempotent: build_features(build_features(df)) == build_features(df)
      (no double-counting of engineered columns).
    - No state: no class with fit/transform -- all features are derivable
      from a single row, no train/test leakage risk.
    - Sklearn-compatible: the bundle returns a DataFrame so downstream
      pipelines (StandardScaler / OneHotEncoder / XGBoost) can consume it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Engagement features
# ---------------------------------------------------------------------
def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engagement-trend and intensity features from raw watch hours."""
    out = df.copy()

    # Trend ratio: scale 7d up to 30-day equivalent, then divide by 30d total.
    # Values < 1 = declining, > 1 = growing, ≈ 1 = stable.
    # Clip to [0, 3] to bound the feature for tree splits.
    out["watch_trend_7d_to_30d"] = (
        (out["watch_hours_last_7d"] * (30 / 7)) /
        out["watch_hours_last_30d"].clip(lower=0.1)
    ).clip(upper=3.0)

    # Same trend but 30d vs 90d -- captures slower-burn decline that the
    # 7d signal would miss. Stable users land at ~1.0, declining < 1, growing > 1.
    out["watch_trend_30d_to_90d"] = (
        (out["watch_hours_last_30d"] * 3) /
        out["watch_hours_last_90d"].clip(lower=0.1)
    ).clip(upper=3.0)

    # Engagement intensity: hours per active login. Captures "do they
    # watch deeply when they show up, or just check in?"
    out["watch_per_login_30d"] = (
        out["watch_hours_last_30d"] /
        out["logins_last_30d"].clip(lower=1)
    ).clip(upper=20.0)

    # Title-breadth proxy: how spread is consumption?
    # High = catalog explorer; low = binge-one-show.
    out["titles_per_hour_30d"] = (
        out["distinct_titles_30d"] /
        out["watch_hours_last_30d"].clip(lower=0.1)
    ).clip(upper=2.0)

    return out


# ---------------------------------------------------------------------
# 2. Tenure features
# ---------------------------------------------------------------------
TENURE_BIN_EDGES = [-1, 1, 2, 5, 11, 13, 24, 60]
TENURE_BIN_LABELS = [
    "m0-1", "m2_trial", "m3-5", "m6-11",
    "m12-13_anniv", "m14-24", "m25_plus",
]


def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure to match the m2 and m12 hazard spikes from Phase 2."""
    out = df.copy()

    out["tenure_bucket"] = pd.cut(
        out["tenure_months"],
        bins=TENURE_BIN_EDGES,
        labels=TENURE_BIN_LABELS,
    )

    # Boolean spike-window flags -- linear model fallback for the bucket.
    out["is_trial_drop_window"] = (
        (out["tenure_months"] >= 2) & (out["tenure_months"] <= 3)
    )
    out["is_anniversary_window"] = (
        (out["tenure_months"] >= 11) & (out["tenure_months"] <= 13)
    )

    return out


# ---------------------------------------------------------------------
# 3. Recency-ratio features
# ---------------------------------------------------------------------
def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-window event counts into recency ratios.

    A recency ratio answers: 'of all events in the long window, what
    fraction happened in the most recent sub-window?' -- a strong signal
    of escalating risk that the raw counts hide.
    """
    out = df.copy()

    # Support-ticket recency: tickets_7d / tickets_90d.
    # Values near 1.0 mean the user's tickets are concentrated in the
    # last week (recent escalation).
    out["tickets_recency_ratio"] = (
        out["support_tickets_7d"] /
        out["support_tickets_90d"].clip(lower=1)
    )

    # Payment-failure recency: failures_30d / failures_180d.
    # Same logic but at 30d-vs-180d horizons.
    out["payment_failures_recency_ratio"] = (
        out["payment_failures_30d"] /
        out["payment_failures_180d"].clip(lower=1)
    )

    # Login recency (recency-density combo)
    out["logins_per_day_30d"] = out["logins_last_30d"] / 30.0

    return out


# ---------------------------------------------------------------------
# 4. Lifecycle features
# ---------------------------------------------------------------------
def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert continuous days-since/days-until lifecycle columns into
    interpretable risk windows."""
    out = df.copy()

    # Recent downgrade flag: plan changed in last 30 days (peak risk window)
    out["recent_plan_change_flag"] = (
        (out["days_since_plan_change"] >= 0) &
        (out["days_since_plan_change"] <= 30)
    )

    # Promo expiring within 14 days
    out["promo_expiring_soon_flag"] = (
        (out["days_until_promo_expires"] >= 0) &
        (out["days_until_promo_expires"] <= 14)
    )

    # Continuous risk score: smooth decay from day 0 to day 90 for plan change.
    # Allows tree splits anywhere in the curve.
    days_since = out["days_since_plan_change"].clip(lower=-1)
    out["plan_change_risk_score"] = np.where(
        (days_since >= 0) & (days_since <= 90),
        1.0 - (days_since / 90.0),
        0.0,
    )

    # Same for promo expiration (1.0 = expiring tomorrow, 0.0 = expired/no promo)
    days_until = out["days_until_promo_expires"].clip(lower=-1)
    out["promo_expiry_risk_score"] = np.where(
        (days_until >= 0) & (days_until <= 30),
        1.0 - (days_until / 30.0),
        0.0,
    )

    return out


# ---------------------------------------------------------------------
# 5. Composite features
# ---------------------------------------------------------------------
def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted interaction features that the Phase 2 heatmap surfaced
    as high-information cells."""
    out = df.copy()

    # The 15.2% churn cell from the Phase 2 heatmap.
    # Used as both a model feature AND a fallback rule for the Retention
    # team when the model's confidence is low.
    out["high_risk_segment_flag"] = (
        out["tenure_bucket"].isin(["m2_trial"]) &
        (out["engagement_cohort"] == "casual")
    )

    # Payment-health composite -- combines method risk + recent failures.
    # Gift-card users with recent failures are the worst-payer cohort.
    out["payment_health_score"] = (
        (out["payment_method"] == "gift_card").astype(int) +
        2 * (out["payment_failures_30d"] > 0).astype(int) +
        (out["auto_renew"] == False).astype(int)
    )

    # Lifecycle-event burden: count of currently-active risk windows
    out["active_risk_event_count"] = (
        out["recent_plan_change_flag"].astype(int) +
        out["promo_expiring_soon_flag"].astype(int) +
        (out["support_tickets_7d"] > 0).astype(int) +
        (out["payment_failures_30d"] > 0).astype(int)
    )

    return out


# ---------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all five feature groups in order. Returns a new DataFrame
    with the original columns plus engineered ones."""
    out = df.copy()
    out = add_engagement_features(out)
    out = add_tenure_features(out)
    out = add_recency_features(out)
    out = add_lifecycle_features(out)
    out = add_composite_features(out)
    return out


# Convenience: which columns are engineered (for downstream selection)
ENGINEERED_COLUMNS = [
    # engagement
    "watch_trend_7d_to_30d", "watch_trend_30d_to_90d",
    "watch_per_login_30d", "titles_per_hour_30d",
    # tenure
    "tenure_bucket", "is_trial_drop_window", "is_anniversary_window",
    # recency
    "tickets_recency_ratio", "payment_failures_recency_ratio",
    "logins_per_day_30d",
    # lifecycle
    "recent_plan_change_flag", "promo_expiring_soon_flag",
    "plan_change_risk_score", "promo_expiry_risk_score",
    # composite
    "high_risk_segment_flag", "payment_health_score",
    "active_risk_event_count",
]
