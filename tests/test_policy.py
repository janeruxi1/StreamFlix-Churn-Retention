"""Decision-rule policy tests. Fully synthetic, no model required."""
import numpy as np
import pandas as pd
import pytest

from src.decisions.policy import (
    INTERVENTION_MENU, LTV_BY_TIER, PREMIUM_UPGRADE_CAP_PCT,
    expected_value, score_all_levers, pick_best_lever,
    apply_budget_cap, apply_premium_cap, summarize_policy,
    simulate_blanket_campaign,
)


# ---- expected_value math ------------------------------------------------
def test_expected_value_formula():
    """EV = p * uplift * LTV - cost, verified on a known example."""
    p = np.array([0.20])
    ltv = np.array([100.0])
    ev = expected_value(p, ltv, uplift=0.10, cost=1.0)[0]
    # 0.20 * 0.10 * 100 - 1 = 2 - 1 = 1
    assert ev == pytest.approx(1.0)


def test_expected_value_negative_when_cost_exceeds_benefit():
    p = np.array([0.01])
    ltv = np.array([50.0])
    ev = expected_value(p, ltv, uplift=0.05, cost=5.0)[0]
    # 0.01 * 0.05 * 50 - 5 = 0.025 - 5 = -4.975
    assert ev < 0


def test_expected_value_scales_linearly_with_ltv():
    p = np.array([0.10])
    ev1 = expected_value(p, np.array([100.0]), uplift=0.10, cost=1.0)[0]
    ev2 = expected_value(p, np.array([200.0]), uplift=0.10, cost=1.0)[0]
    # ev1 = 1 - 1 = 0; ev2 = 2 - 1 = 1
    assert ev2 - ev1 == pytest.approx(1.0)


# ---- score_all_levers ---------------------------------------------------
def test_score_all_levers_returns_one_column_per_lever(small_probas, small_ltv):
    result = score_all_levers(small_probas, small_ltv, INTERVENTION_MENU)
    for lever in INTERVENTION_MENU:
        assert f"ev_{lever}" in result.columns


def test_score_all_levers_row_count_matches_input(small_probas, small_ltv):
    result = score_all_levers(small_probas, small_ltv, INTERVENTION_MENU)
    assert len(result) == len(small_probas)


# ---- pick_best_lever ----------------------------------------------------
def test_pick_best_lever_picks_none_for_low_risk_users(small_probas, small_ltv):
    """Users with 2% churn prob shouldn't be worth any lever."""
    result = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    assert result.iloc[0]["best_lever"] == "none"
    assert result.iloc[0]["best_ev"] == 0.0
    assert result.iloc[0]["cost"] == 0.0


def test_pick_best_lever_picks_a_lever_for_high_risk_users(small_probas, small_ltv):
    """A user at 90% churn prob with $228 LTV should have positive EV levers."""
    result = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    assert result.iloc[-1]["best_lever"] != "none"
    assert result.iloc[-1]["best_ev"] > 0


def test_pick_best_lever_shape(small_probas, small_ltv):
    result = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    assert len(result) == len(small_probas)
    assert {"best_lever", "best_ev", "cost"} <= set(result.columns)


# ---- apply_budget_cap ---------------------------------------------------
def test_apply_budget_cap_respects_budget(small_probas, small_ltv):
    policy = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    capped = apply_budget_cap(policy, budget=5.0)
    total_cost = capped.loc[capped["will_target"], "cost"].sum()
    assert total_cost <= 5.0


def test_apply_budget_cap_zero_budget_targets_none(small_probas, small_ltv):
    policy = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    capped = apply_budget_cap(policy, budget=0.0)
    assert not capped["will_target"].any()


def test_apply_budget_cap_orders_by_ev(small_probas, small_ltv):
    """With enough budget, highest-EV users must be targeted first."""
    policy = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    capped = apply_budget_cap(policy, budget=1_000_000.0)
    actionable = policy[policy["best_lever"] != "none"]
    if len(actionable) > 0:
        # All positive-EV users should be targeted at unlimited budget
        assert capped.loc[actionable.index, "will_target"].all()


# ---- apply_premium_cap --------------------------------------------------
def test_apply_premium_cap_reduces_premium_count():
    """If too many premium recommendations, cap kicks in."""
    n = 100
    df = pd.DataFrame({
        "best_lever": ["premium_upgrade"] * n,
        "best_ev": np.linspace(10, 1, n),
        "cost": [12.0] * n,
        "will_target": [True] * n,
    })
    capped = apply_premium_cap(df, n_total=100, cap_pct=0.05)
    # After the cap: only 5 users retain 'premium_upgrade'
    remaining = (capped["best_lever"] == "premium_upgrade").sum()
    assert remaining == 5


def test_apply_premium_cap_no_op_when_under_cap():
    """If premium count is already below the cap, no changes."""
    df = pd.DataFrame({
        "best_lever": ["credit_5", "premium_upgrade", "credit_5"],
        "best_ev": [3.0, 2.0, 1.0],
        "cost": [5.0, 12.0, 5.0],
        "will_target": [True, True, True],
    })
    capped = apply_premium_cap(df.copy(), n_total=1000, cap_pct=0.05)
    # 1 premium out of 1000 subscribers is under the 5% (=50) cap
    pd.testing.assert_frame_equal(capped, df)


# ---- summarize_policy ---------------------------------------------------
def test_summarize_policy_math(small_probas, small_ltv):
    """net_ev should equal revenue - cost."""
    policy = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    policy = apply_budget_cap(policy, budget=1_000_000.0)
    summary = summarize_policy(policy, targeted_only=True)
    revenue = summary["expected_retained_revenue"]
    cost = summary["total_cost"]
    net = summary["net_expected_value"]
    assert net == pytest.approx(revenue - cost, abs=1e-6)


def test_summarize_policy_empty_targeting(small_probas, small_ltv):
    """No targeted users -> zero cost, zero revenue, ROI = 0."""
    policy = pick_best_lever(small_probas, small_ltv, INTERVENTION_MENU)
    policy["will_target"] = False
    summary = summarize_policy(policy, targeted_only=True)
    assert summary["n_targeted"] == 0
    assert summary["total_cost"] == 0.0
    assert summary["roi_multiplier"] == 0.0


# ---- simulate_blanket_campaign -----------------------------------------
def test_blanket_campaign_only_targets_specified_month():
    n = 100
    p_churn = np.full(n, 0.10)
    ltv = np.full(n, 100.0)
    tenure = np.arange(n) % 24  # tenure 0..23
    result = simulate_blanket_campaign(
        p_churn, ltv, tenure, cost_per_user=5.0,
        target_month=11, uplift=0.15,
    )
    # Users with tenure == 11 get targeted
    n_target = (tenure == 11).sum()
    assert result["n_targeted"] == n_target


def test_blanket_campaign_cost_matches_users_times_price():
    n = 50
    tenure = np.full(n, 11)
    result = simulate_blanket_campaign(
        np.full(n, 0.10), np.full(n, 100.0), tenure,
        cost_per_user=5.0, target_month=11, uplift=0.15,
    )
    assert result["total_cost"] == 50 * 5.0
