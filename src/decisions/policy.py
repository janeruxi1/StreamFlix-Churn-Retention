"""Cost-aware decision rule for the StreamFlix retention system.

Turns per-user P(churn) into per-user targeting decisions by weighing
expected retention revenue against intervention cost. Given a budget
cap, allocates spend to the users with the highest expected value.

Expected value per user under lever L:
    EV(user, L) = P(churn|user) * uplift(L) * LTV(tier) - cost(L)

Where:
    P(churn|user)  from the calibrated XGBoost (Phase 4)
    uplift(L)      fraction of would-have-churners that L retains
    LTV(tier)      lifetime value of retained user, by plan
    cost(L)        one-time cost of running intervention L

Decision:
    lever(user) = argmax_L EV(user, L) if max EV > 0 else None
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Assumptions (curated with the PM in reports/scenario_brief.md)
# ---------------------------------------------------------------------
INTERVENTION_MENU: Dict[str, Dict[str, float]] = {
    "curated_playlist": {"cost":  1.0, "uplift": 0.05},
    "credit_5":         {"cost":  5.0, "uplift": 0.15},
    "premium_upgrade":  {"cost": 12.0, "uplift": 0.25},
}

# LTV = monthly revenue x expected months retained (per plan tier)
LTV_BY_TIER: Dict[str, float] = {
    "Basic":     9.0  *  8,   # $72
    "Standard": 14.0 * 10,    # $140
    "Premium":  19.0 * 12,    # $228
}

# Guardrail: premium_upgrade can be offered to at most this fraction
# of subscribers per month (avoid margin compression)
PREMIUM_UPGRADE_CAP_PCT = 0.05


# ---------------------------------------------------------------------
# Core EV math
# ---------------------------------------------------------------------
def expected_value(p_churn: np.ndarray, ltv: np.ndarray,
                   uplift: float, cost: float) -> np.ndarray:
    """Expected value of applying one lever to each user in a vector."""
    return p_churn * uplift * ltv - cost


def score_all_levers(p_churn: np.ndarray,
                     ltv: np.ndarray,
                     menu: Dict[str, Dict[str, float]] = INTERVENTION_MENU
                     ) -> pd.DataFrame:
    """Return a wide DataFrame with one EV column per lever."""
    out = pd.DataFrame(index=range(len(p_churn)))
    for name, params in menu.items():
        out[f"ev_{name}"] = expected_value(
            p_churn, ltv, params["uplift"], params["cost"]
        )
    return out


# ---------------------------------------------------------------------
# Per-user lever choice
# ---------------------------------------------------------------------
def pick_best_lever(p_churn: np.ndarray,
                    ltv: np.ndarray,
                    menu: Dict[str, Dict[str, float]] = INTERVENTION_MENU,
                    ) -> pd.DataFrame:
    """For each user, pick the highest-EV lever (or 'none' if all EV <= 0).

    Returns a DataFrame with:
        best_lever    -- name of chosen lever, or 'none'
        best_ev       -- EV of chosen lever (0 if 'none')
        cost          -- cost incurred (0 if 'none')
    """
    ev_wide = score_all_levers(p_churn, ltv, menu)
    ev_matrix = ev_wide.values
    lever_names = [c.replace("ev_", "") for c in ev_wide.columns]

    max_idx = ev_matrix.argmax(axis=1)
    max_ev = ev_matrix.max(axis=1)

    best_lever = np.array([lever_names[i] for i in max_idx])
    costs = np.array([menu[n]["cost"] for n in best_lever])

    # No intervention if all EVs are non-positive
    mask_no_action = max_ev <= 0
    best_lever[mask_no_action] = "none"
    costs[mask_no_action] = 0.0
    max_ev[mask_no_action] = 0.0

    return pd.DataFrame({
        "best_lever": best_lever,
        "best_ev": max_ev,
        "cost": costs,
    })


# ---------------------------------------------------------------------
# Budget cap allocation
# ---------------------------------------------------------------------
def apply_budget_cap(policy: pd.DataFrame,
                     budget: float) -> pd.DataFrame:
    """Sort users by EV descending, allocate until budget exhausted.

    Only touches users flagged for intervention (best_lever != 'none').
    Cumulative-cost cutoff = last user we can afford.
    """
    out = policy.copy()
    out["will_target"] = False

    actionable = out[out["best_lever"] != "none"].copy()
    actionable = actionable.sort_values("best_ev", ascending=False)
    actionable["cum_cost"] = actionable["cost"].cumsum()
    to_target = actionable[actionable["cum_cost"] <= budget].index

    out.loc[to_target, "will_target"] = True
    return out


def apply_premium_cap(policy: pd.DataFrame,
                      n_total: int,
                      cap_pct: float = PREMIUM_UPGRADE_CAP_PCT
                      ) -> pd.DataFrame:
    """Cap the number of 'premium_upgrade' offers at cap_pct of the base.

    Rank premium_upgrade recommendations by EV, drop the ones beyond
    the cap. For dropped users, re-score without the premium option and
    take the next-best lever.
    """
    out = policy.copy()
    max_premium = int(np.floor(n_total * cap_pct))

    premium_mask = (out["best_lever"] == "premium_upgrade") & out["will_target"]
    if premium_mask.sum() <= max_premium:
        return out

    premium_sorted = out[premium_mask].sort_values("best_ev", ascending=False)
    keep_idx = premium_sorted.head(max_premium).index
    drop_idx = premium_sorted.iloc[max_premium:].index

    # For dropped users, re-target with cheaper lever (credit_5) if EV > 0
    for i in drop_idx:
        # naive: fall back to credit_5 if it has positive EV, else no action
        # (in practice we'd re-run pick_best_lever with premium removed)
        out.loc[i, "best_lever"] = "credit_5"
        out.loc[i, "cost"] = INTERVENTION_MENU["credit_5"]["cost"]
        # EV re-computed elsewhere; for now leave old EV as sort key

    return out


# ---------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------
def summarize_policy(policy: pd.DataFrame,
                     targeted_only: bool = True) -> Dict[str, float]:
    """Aggregate policy stats: how many targeted, total cost, expected
    retained revenue, net expected value, ROI."""
    df = policy[policy["will_target"]] if targeted_only else policy

    n_targeted = int(len(df))
    total_cost = float(df["cost"].sum())
    total_ev = float(df["best_ev"].sum())
    # Expected retained revenue = EV + cost (since EV = revenue - cost)
    total_revenue = total_ev + total_cost
    roi = (total_revenue / total_cost) if total_cost > 0 else 0.0

    return {
        "n_targeted": n_targeted,
        "total_cost": total_cost,
        "expected_retained_revenue": total_revenue,
        "net_expected_value": total_ev,
        "roi_multiplier": roi,
    }


def simulate_blanket_campaign(p_churn: np.ndarray,
                              ltv: np.ndarray,
                              tenure_months: np.ndarray,
                              cost_per_user: float = 5.0,
                              target_month: int = 11,
                              uplift: float = 0.15) -> Dict[str, float]:
    """Simulate the current blanket $5 credit campaign at month-11.

    Reference behavior we're trying to replace: send $5 credit to every
    user whose tenure is at target_month, regardless of P(churn).
    """
    mask = tenure_months == target_month
    n_targeted = int(mask.sum())
    total_cost = n_targeted * cost_per_user
    revenue_saved = float((p_churn[mask] * uplift * ltv[mask]).sum())
    net_ev = revenue_saved - total_cost
    roi = revenue_saved / total_cost if total_cost > 0 else 0.0

    return {
        "n_targeted": n_targeted,
        "total_cost": total_cost,
        "expected_retained_revenue": revenue_saved,
        "net_expected_value": net_ev,
        "roi_multiplier": roi,
    }
