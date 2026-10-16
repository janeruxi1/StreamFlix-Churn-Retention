"""Cost-aware decision rule for the StreamFlix retention system.

Turns per-user P(churn) into per-user targeting decisions by weighing
expected retention revenue against intervention cost. Given a budget
cap, allocates spend to the users with the highest expected value.

Expected value per user under lever L:
    EV(user, L) = P(churn|user) * uplift(L) * LTV(tier) - cost(L)

Where:
    P(churn|user)  from the calibrated HistGradientBoosting (Phase 4)
    uplift(L)      fraction of would-have-churners that L retains
    LTV(tier)      lifetime value of retained user, by plan
    cost(L)        one-time cost of running intervention L

Decision:
    lever(user) = argmax_L EV(user, L) if max EV > 0 else None
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Re-exported on purpose: app/streamlit_app.py and notebooks 05 and 08
# import LTV_BY_TIER from this module. Removing it breaks the live demo.
from src.decisions.ltv import LTV_BY_TIER  # noqa: F401


# ---------------------------------------------------------------------
# Assumptions (curated with the PM in reports/scenario_brief.md)
# ---------------------------------------------------------------------
INTERVENTION_MENU: Dict[str, Dict[str, float]] = {
    "curated_playlist": {"cost":  1.0, "uplift": 0.05},
    "credit_5":         {"cost":  5.0, "uplift": 0.15},
    "premium_upgrade":  {"cost": 12.0, "uplift": 0.25},
}

# LTV_BY_TIER imported from src/decisions/ltv.py -- derived from Phase 2's
# survival curves via restricted mean survival time (24-month horizon).

# ---------------------------------------------------------------------
# Scale: the brief's budget and caps are for the whole subscriber base,
# but the analysis runs on a 50,000-subscriber sample. Every money or
# capacity limit is scaled to the sample before it is applied.
# ---------------------------------------------------------------------
POPULATION_SUBSCRIBERS = 2_100_000
MONTHLY_BUDGET = 200_000.0           # $ per month, whole base (brief)
PREMIUM_OFFERS_PER_MONTH = 10_000    # ops capacity, whole base (brief)

# Guardrail: premium_upgrade can be offered to at most this share of
# subscribers per month (10k of 2.1M = 0.48%).
PREMIUM_UPGRADE_CAP_PCT = PREMIUM_OFFERS_PER_MONTH / POPULATION_SUBSCRIBERS


def sample_budget(n_sample: int,
                  monthly_budget: float = MONTHLY_BUDGET,
                  population: int = POPULATION_SUBSCRIBERS) -> float:
    """The monthly budget scaled to a sample of `n_sample` subscribers."""
    return monthly_budget * n_sample / population


def to_population(amount: float, n_sample: int,
                  population: int = POPULATION_SUBSCRIBERS) -> float:
    """Scale a sample-level count or dollar amount up to the whole base."""
    return amount * population / n_sample


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

    best_lever = np.array([lever_names[i] for i in max_idx], dtype=object)
    costs = np.array([menu[n]["cost"] for n in best_lever])

    # No intervention if all EVs are non-positive
    mask_no_action = max_ev <= 0
    best_lever[mask_no_action] = "none"
    costs[mask_no_action] = 0.0
    max_ev[mask_no_action] = 0.0

    out = pd.DataFrame({
        "best_lever": best_lever,
        "best_ev": max_ev,
        "cost": costs,
    })
    # Keep every lever's EV so a capped lever can fall back to the next best
    return pd.concat([out, ev_wide], axis=1)


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

    Rank premium_upgrade recommendations by EV and keep the top ones up
    to the cap. Users beyond the cap fall back to their best remaining
    lever if its EV is positive (needs the per-lever `ev_*` columns that
    pick_best_lever returns), otherwise to no action. Apply this BEFORE
    apply_budget_cap so the budget is spent on the final lever mix.
    """
    out = policy.copy()
    max_premium = int(np.floor(n_total * cap_pct))

    premium_mask = out["best_lever"] == "premium_upgrade"
    if "will_target" in out.columns:
        premium_mask &= out["will_target"]
    if premium_mask.sum() <= max_premium:
        return out

    premium_sorted = out[premium_mask].sort_values("best_ev", ascending=False)
    drop_idx = premium_sorted.iloc[max_premium:].index

    fallback = [c for c in out.columns
                if c.startswith("ev_") and c != "ev_premium_upgrade"]
    for i in drop_idx:
        evs = out.loc[i, fallback].astype(float) if fallback else pd.Series(dtype=float)
        if len(evs) and evs.max() > 0:
            lever = evs.idxmax().replace("ev_", "")
            out.loc[i, ["best_lever", "best_ev", "cost"]] = [
                lever, float(evs.max()), INTERVENTION_MENU[lever]["cost"]]
        else:
            out.loc[i, ["best_lever", "best_ev", "cost"]] = ["none", 0.0, 0.0]
            if "will_target" in out.columns:
                out.loc[i, "will_target"] = False

    return out


def _choose(ev: np.ndarray, cost: np.ndarray, lam: float,
            allowed: np.ndarray) -> np.ndarray:
    """Per user, the lever index maximising EV - lam * cost (-1 = none)."""
    adj = np.where(allowed, ev - lam * cost[None, :], -np.inf)
    j = adj.argmax(axis=1)
    j[adj[np.arange(len(j)), j] <= 0] = -1
    return j


def build_policy(p_churn: np.ndarray,
                 ltv: np.ndarray,
                 budget: float,
                 menu: Dict[str, Dict[str, float]] = INTERVENTION_MENU,
                 premium_cap_pct: float = PREMIUM_UPGRADE_CAP_PCT,
                 ) -> pd.DataFrame:
    """Budget-constrained v1 policy (one lever or none per user).

    With a binding budget, ranking users by raw EV is not optimal: a $1
    lever earning $0.80 beats a $5 lever earning $2 per dollar spent. The
    right rule charges every dollar its opportunity cost lam (a Lagrange
    multiplier): each user gets the lever maximising EV - lam * cost, if
    that is positive, and lam is the smallest value whose total spend fits
    the budget. lam = 0 when the budget doesn't bind (plain argmax EV).
    The premium_upgrade capacity cap is enforced by moving the
    lowest-value premium users to their best remaining lever.

    Returns the same columns as pick_best_lever plus `will_target`.
    """
    names = list(menu)
    cost = np.array([menu[n]["cost"] for n in names])
    uplift = np.array([menu[n]["uplift"] for n in names])
    p = np.asarray(p_churn, dtype=float)
    ev = p[:, None] * uplift[None, :] * np.asarray(ltv, dtype=float)[:, None] - cost[None, :]
    allowed = np.ones_like(ev, dtype=bool)
    max_premium = int(np.floor(len(p) * premium_cap_pct))
    prem = names.index("premium_upgrade") if "premium_upgrade" in names else None

    def spend(lam: float) -> float:
        j = _choose(ev, cost, lam, allowed)
        return float(cost[j[j >= 0]].sum())

    for _ in range(3):  # premium cap can shift spend; re-solve a few times
        lo, hi = 0.0, 1.0
        if spend(0.0) > budget:
            while spend(hi) > budget:
                hi *= 2
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if spend(mid) > budget else (lo, mid)
            lam = hi
        else:
            lam = 0.0
        j = _choose(ev, cost, lam, allowed)
        if prem is None or (j == prem).sum() <= max_premium:
            break
        prem_users = np.where(j == prem)[0]
        order = prem_users[np.argsort(-(ev[prem_users, prem] - lam * cost[prem]))]
        allowed[order[max_premium:], prem] = False

    j = _choose(ev, cost, lam, allowed)
    # Bisection leaves spend <= budget; drop the least valuable users if a
    # tie pushed it over.
    chosen = np.where(j >= 0)[0]
    total = cost[j[chosen]].sum()
    if total > budget:
        val = ev[chosen, j[chosen]] - lam * cost[j[chosen]]
        for i in chosen[np.argsort(val)]:
            if total <= budget:
                break
            total -= cost[j[i]]
            j[i] = -1

    target = j >= 0
    best_lever = np.where(target, np.array(names, dtype=object)[np.maximum(j, 0)], "none")
    best_ev = np.where(target, ev[np.arange(len(p)), np.maximum(j, 0)], 0.0)
    out = pd.DataFrame({
        "best_lever": best_lever,
        "best_ev": best_ev,
        "cost": np.where(target, cost[np.maximum(j, 0)], 0.0),
        "will_target": target,
    })
    out = pd.concat([out, pd.DataFrame(ev, columns=[f"ev_{n}" for n in names])], axis=1)
    out.attrs["shadow_price"] = lam  # $ of EV forgone per extra $ of budget
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


# ---------------------------------------------------------------------
# Uplift-based lever selection (Phase 8)
# ---------------------------------------------------------------------
def pick_best_lever_uplift(uplift_by_lever: Dict[str, np.ndarray],
                           ltv: np.ndarray,
                           menu: Dict[str, Dict[str, float]] = INTERVENTION_MENU,
                           ) -> pd.DataFrame:
    """Uplift-aware version of pick_best_lever.

    Instead of `P(churn) * uplift_constant * LTV - cost`, this uses the
    per-user LEARNED retention lift from an uplift model:

        EV(user, L) = retention_lift(user, L) * LTV(tier) - cost(L)

    Where `retention_lift(user, L) = P(churn|control) - P(churn|treated)`,
    which is the sign-flipped output of a scikit-uplift model.

    Advantages over the fixed-uplift version:
      - Catches "persuadables" that a P(churn)-only model would miss
        because their absolute churn probability isn't high, but
        treatment shifts them a lot.
      - Skips "sleeping dogs" — users with negative retention lift
        (treatment makes them more likely to churn). Fixed-uplift
        pulls them in whenever P(churn) is high; uplift-based excludes
        them because their per-user lift is negative.

    Parameters:
        uplift_by_lever: {lever_name: per_user_retention_lift_array}.
                         Missing levers are skipped. If passed with only
                         one lever, the returned policy uses that lever
                         exclusively (with a "none" fallback for users
                         whose EV goes non-positive).
        ltv:             per-user LTV.
        menu:            intervention menu (for costs).

    Returns:
        DataFrame with columns best_lever, best_ev, cost.
    """
    lever_names = list(uplift_by_lever.keys())
    if not lever_names:
        raise ValueError("uplift_by_lever must contain at least one lever")

    ev_matrix = np.column_stack([
        uplift_by_lever[name] * ltv - menu[name]["cost"]
        for name in lever_names
    ])
    max_idx = ev_matrix.argmax(axis=1)
    max_ev = ev_matrix.max(axis=1)
    best_lever = np.array([lever_names[i] for i in max_idx])
    costs = np.array([menu[n]["cost"] for n in best_lever])

    # No intervention if best EV <= 0 (including sleeping dogs with
    # negative uplift on every lever)
    no_action = max_ev <= 0
    best_lever[no_action] = "none"
    costs[no_action] = 0.0
    max_ev[no_action] = 0.0

    return pd.DataFrame({
        "best_lever": best_lever,
        "best_ev": max_ev,
        "cost": costs,
    })


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
