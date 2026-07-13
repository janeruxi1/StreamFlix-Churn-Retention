"""Simulator invariants -- schema, reproducibility, sane ranges,
and the nesting properties we rely on downstream."""
import numpy as np
import pandas as pd

from src.data.simulate import simulate_subscribers, SimConfig
from src.data.loader import EXPECTED_COLUMNS


def test_simulate_reproducible():
    """Same seed -> same rows."""
    a = simulate_subscribers(SimConfig(n_subscribers=500, seed=42))
    b = simulate_subscribers(SimConfig(n_subscribers=500, seed=42))
    pd.testing.assert_frame_equal(a, b)


def test_simulate_different_seeds_diverge():
    """Different seeds -> different data."""
    a = simulate_subscribers(SimConfig(n_subscribers=500, seed=42))
    b = simulate_subscribers(SimConfig(n_subscribers=500, seed=123))
    assert not a.equals(b)


def test_simulate_row_count():
    df = simulate_subscribers(SimConfig(n_subscribers=1234, seed=42))
    assert len(df) == 1234


def test_expected_columns_present(raw_df):
    missing = EXPECTED_COLUMNS - set(raw_df.columns)
    assert not missing, f"missing columns: {missing}"


def test_churn_rate_in_realistic_band(raw_df):
    """Overall 30-day churn should land in the 3-10% band."""
    rate = raw_df["churned_next_30d"].mean()
    assert 0.03 <= rate <= 0.10, f"churn rate {rate:.2%} outside 3-10% band"


def test_no_null_columns(raw_df):
    """Simulator should never produce NaNs in any column."""
    nulls = raw_df.isnull().sum()
    assert nulls.sum() == 0, f"null columns: {nulls[nulls > 0]}"


def test_watch_hours_multi_window_consistent(raw_df):
    """watch_30d * 3 ~ watch_90d on average, and both > 0.

    The bug we caught in Phase 3 debugging -- watch_90d was set equal
    to watch_30d instead of scaled to 3x. This test prevents regressions.
    """
    r30 = raw_df["watch_hours_last_30d"].mean() * 3
    r90 = raw_df["watch_hours_last_90d"].mean()
    ratio = r30 / max(r90, 0.1)
    assert 0.85 <= ratio <= 1.15, (
        f"watch_30d * 3 / watch_90d = {ratio:.3f}, expected near 1.0"
    )


def test_support_ticket_windows_nested(raw_df):
    """tickets_7d <= tickets_30d <= tickets_90d for every user (Poisson thinning)."""
    assert (raw_df["support_tickets_7d"] <= raw_df["support_tickets_30d"]).all()
    assert (raw_df["support_tickets_30d"] <= raw_df["support_tickets_90d"]).all()


def test_payment_failure_windows_nested(raw_df):
    """failures_30d <= failures_90d <= failures_180d for every user."""
    assert (raw_df["payment_failures_30d"] <= raw_df["payment_failures_90d"]).all()
    assert (raw_df["payment_failures_90d"] <= raw_df["payment_failures_180d"]).all()


def test_categorical_values_in_expected_sets(raw_df):
    assert set(raw_df["plan_tier"].unique()) <= {"Basic", "Standard", "Premium"}
    assert set(raw_df["billing_cycle"].unique()) <= {"monthly", "annual"}
    assert set(raw_df["engagement_cohort"].unique()) <= {"heavy", "regular", "casual"}
    assert set(raw_df["payment_method"].unique()) <= {"credit_card", "paypal", "gift_card"}


def test_tenure_range(raw_df):
    """tenure_months should be a non-negative integer within a reasonable band."""
    assert raw_df["tenure_months"].min() >= 0
    assert raw_df["tenure_months"].max() <= 60
    assert raw_df["tenure_months"].dtype in (np.int64, np.int32, int)


def test_lifecycle_sentinels(raw_df):
    """days_since_plan_change and days_until_promo_expires use -1 = 'never/none'."""
    assert (raw_df["days_since_plan_change"] >= -1).all()
    assert (raw_df["days_until_promo_expires"] >= -1).all()
