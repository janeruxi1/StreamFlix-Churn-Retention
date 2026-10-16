"""Tests for src/decisions/ltv.py -- LTV derivation from survival curves.

If the hardcoded LTV_BY_TIER constants in ltv.py drift from what the
derivation actually produces on the current dataset, these tests fail.
"""
import numpy as np

from src.decisions.ltv import (
    LTV_BY_TIER, MONTHLY_REVENUE_BY_TIER, RMST_HORIZON_MONTHS,
    life_table_survival, restricted_mean_survival_time, derive_ltv_from_data,
)


def test_life_table_uses_users_at_each_tenure_as_risk_set():
    """Month-t hazard = churners at tenure t / users at tenure t.

    Tenure 0: 4 users, 1 churns -> h = 0.25.  Tenure 1: no users -> h = 0.
    Tenure 2: 2 users, 1 churns -> h = 0.5.
    """
    durations = np.array([0, 0, 0, 0, 2, 2])
    events = np.array([1, 0, 0, 0, 1, 0])
    t, S = life_table_survival(durations, events)
    assert list(t) == [1, 2, 3]
    assert np.allclose(S, [0.75, 0.75, 0.375])
    assert (np.diff(S) <= 1e-12).all()


def test_life_table_matches_constant_hazard():
    """With the same 5% churn at every tenure, S(t) = 0.95 ** t."""
    durations = np.repeat(np.arange(24), 100)
    events = np.tile(np.r_[np.ones(5), np.zeros(95)], 24)
    _, S = life_table_survival(durations, events)
    assert np.allclose(S, 0.95 ** np.arange(1, 25))


def test_rmst_of_constant_1_is_horizon():
    """If no one ever churns (S(t) = 1 forever), RMST = horizon."""
    t = np.array([], dtype=float)
    S = np.array([], dtype=float)
    assert abs(restricted_mean_survival_time(t, S, horizon=24) - 24) < 1e-9


def test_rmst_of_instant_death_is_zero():
    """If everyone churns at t=0.5, RMST(24) ≈ 0.5."""
    t = np.array([0.5])
    S = np.array([0.0])
    assert abs(restricted_mean_survival_time(t, S, horizon=24) - 0.5) < 1e-9


def test_rmst_step_function_area():
    """S drops from 1 to 0.5 at t=10, then 0.5 to 0 at t=20. Horizon 24.
    Area = 1*10 + 0.5*10 + 0*4 = 15."""
    t = np.array([10, 20])
    S = np.array([0.5, 0.0])
    assert abs(restricted_mean_survival_time(t, S, horizon=24) - 15.0) < 1e-9


def test_ltv_constants_match_km_derivation(raw_df):
    """The hardcoded LTV_BY_TIER values must match derive_ltv_from_data()
    on the current dataset. If someone changes the simulator or the
    horizon and forgets to update the constants, this catches it."""
    derived = derive_ltv_from_data(raw_df, horizon=RMST_HORIZON_MONTHS)
    for tier, expected in LTV_BY_TIER.items():
        # Loose tolerance: the test fixture is a 2,000-row sample, which is
        # much noisier than the 50k production dataset.
        assert abs(derived[tier] - expected) < 60, (
            f"LTV drift for {tier}: hardcoded ${expected}, "
            f"derived ${derived[tier]} on the current dataset. "
            f"Either update LTV_BY_TIER in src/decisions/ltv.py or fix "
            f"whatever changed in the survival curves."
        )


def test_ltv_ordering_holds(raw_df):
    """Premium > Standard > Basic on LTV -- higher tier retains longer AND
    pays more, so LTV should be monotonic in tier."""
    derived = derive_ltv_from_data(raw_df)
    assert derived["Basic"] < derived["Standard"] < derived["Premium"]


def test_monthly_revenue_matches_simulator():
    """Monthly revenue constants in ltv.py must match the simulator's."""
    from src.data.simulate import SimConfig
    cfg = SimConfig()
    for tier, expected in cfg.monthly_revenue.items():
        assert MONTHLY_REVENUE_BY_TIER[tier] == expected, (
            f"Monthly revenue for {tier} disagrees: "
            f"ltv.py has ${MONTHLY_REVENUE_BY_TIER[tier]}, "
            f"simulator has ${expected}"
        )
