"""Feature engineering transforms -- idempotency, shapes, and value ranges."""
import numpy as np
import pandas as pd
import pytest

from src.features.transforms import (
    build_features, ENGINEERED_COLUMNS,
    add_engagement_features, add_tenure_features,
    add_recency_features, add_lifecycle_features,
    add_composite_features,
)


def test_build_features_adds_all_engineered_columns(raw_df):
    df = build_features(raw_df)
    for col in ENGINEERED_COLUMNS:
        assert col in df.columns, f"missing engineered column: {col}"


def test_build_features_preserves_raw_columns(raw_df):
    """Every original column must still be there after engineering."""
    df = build_features(raw_df)
    for col in raw_df.columns:
        assert col in df.columns


def test_build_features_is_idempotent(raw_df):
    """build_features(build_features(df)) == build_features(df)."""
    once = build_features(raw_df)
    twice = build_features(once)
    pd.testing.assert_frame_equal(once, twice)


def test_build_features_does_not_mutate_input(raw_df):
    """Pure functions: raw_df should be unchanged after engineering."""
    before_shape = raw_df.shape
    before_cols = list(raw_df.columns)
    _ = build_features(raw_df)
    assert raw_df.shape == before_shape
    assert list(raw_df.columns) == before_cols


def test_watch_trend_around_1_for_stable_users(engineered_df):
    """Stable-engagement users should cluster around trend ~= 1.0."""
    med = engineered_df["watch_trend_7d_to_30d"].median()
    assert 0.85 <= med <= 1.15, f"median trend {med:.3f} not near 1.0"


def test_watch_trend_30d_to_90d_not_pinned_at_ceiling(engineered_df):
    """Regression test for the simulator bug we caught in Phase 3.

    Before the fix, ~56% of users hit the 3.0 clip ceiling. After the
    fix, mean should be near 1.0.
    """
    ratio = engineered_df["watch_trend_30d_to_90d"].mean()
    assert 0.85 <= ratio <= 1.15, (
        f"trend mean {ratio:.3f} suggests users pinned at ceiling"
    )
    # Also check: fewer than 20% should hit the 3.0 clip ceiling
    at_ceiling = (engineered_df["watch_trend_30d_to_90d"] >= 2.99).mean()
    assert at_ceiling < 0.20, f"{at_ceiling:.1%} of users at ceiling -- too many"


def test_tenure_bucket_covers_all_users(engineered_df):
    """No user should be uncategorized by the tenure bucketer."""
    assert engineered_df["tenure_bucket"].notna().all()


def test_recency_ratios_in_unit_interval(engineered_df):
    """tickets_recency_ratio and payment_failures_recency_ratio in [0, 1]."""
    for col in ["tickets_recency_ratio", "payment_failures_recency_ratio"]:
        s = engineered_df[col]
        assert (s >= 0).all(), f"{col} has negative values"
        assert (s <= 1).all(), f"{col} has values > 1"


def test_lifecycle_risk_scores_in_unit_interval(engineered_df):
    """plan_change_risk_score and promo_expiry_risk_score in [0, 1]."""
    for col in ["plan_change_risk_score", "promo_expiry_risk_score"]:
        s = engineered_df[col]
        assert (s >= 0).all(), f"{col} has negative values"
        assert (s <= 1).all(), f"{col} has values > 1"


def test_boolean_flags_are_boolean(engineered_df):
    """Engineered flag columns should be boolean or 0/1."""
    for col in ["is_trial_drop_window", "is_anniversary_window",
                "recent_plan_change_flag", "promo_expiring_soon_flag",
                "high_risk_segment_flag"]:
        s = engineered_df[col]
        assert s.dtype == bool or set(s.unique()) <= {0, 1, True, False}, (
            f"{col} is not boolean-typed"
        )


def test_high_risk_segment_flag_matches_definition(engineered_df):
    """high_risk_segment_flag = (tenure_bucket == 'm2_trial') & (cohort == 'casual')."""
    expected = (
        (engineered_df["tenure_bucket"] == "m2_trial") &
        (engineered_df["engagement_cohort"] == "casual")
    )
    actual = engineered_df["high_risk_segment_flag"].astype(bool)
    assert (expected == actual).all()


def test_no_null_introduced_by_transforms(engineered_df):
    """Feature engineering shouldn't create NaN values."""
    nulls = engineered_df[ENGINEERED_COLUMNS].isnull().sum()
    assert nulls.sum() == 0, f"engineered features have nulls: {nulls[nulls > 0]}"


def test_add_engagement_features_standalone(raw_df):
    """Each group function works in isolation."""
    result = add_engagement_features(raw_df)
    assert "watch_trend_7d_to_30d" in result.columns
    assert "watch_trend_30d_to_90d" in result.columns
