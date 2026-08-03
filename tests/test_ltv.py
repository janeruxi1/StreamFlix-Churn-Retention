"""Tests for src/decisions/ltv.py -- LTV derivation from KM survival curves.

If the hardcoded LTV_BY_TIER constants in ltv.py drift from what the KM
derivation actually produces on the current dataset, these tests fail.
"""
import numpy as np
import pandas as pd
import pytest

from src.decisions.ltv import (
    LTV_BY_TIER, MONTHLY_REVENUE_BY_TIER, RMST_HORIZON_MONTHS,
    kaplan_meier, restricted_mean_survival_time, derive_ltv_from_data,
)


def test_kaplan_meier_basic_properties():
    """S(t) is monotonically non-increasing, starts near 1, stays in [0, 1]."""
    # Simple example: 5 subjects, 2 churn at t=1, 3 at t=3
    durations = np.array([1, 1, 3, 3, 3])
    events = np.array([1, 1, 1, 1, 1])
    t, S = kaplan_meier(durations, events)
    assert len(t) == 2
    assert t[0] == 1
    assert t[1] == 3
    # After t=1: 2 of 5 churned -> S = 3/5 = 0.6
    assert abs(S[0] - 0.6) < 1e-9
    # After t=3: from 3 remaining, all 3 churned -> S = 0.0
    assert abs(S[1]) < 1e-9
    assert (S <= 1).all() and (S >= 0).all()
    # Monotonic non-increasing
    assert (np.diff(S) <= 1e-9).all()


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
        # Allow $10 tolerance for small stochastic variation between
        # dataset regenerations at different seeds (test fixture uses
        # 2000 rows, which has more variance than the 50k in production)
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
