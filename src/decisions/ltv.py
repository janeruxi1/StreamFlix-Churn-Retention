"""LTV constants + survival-based derivation for the Phase 6 decision rule.

Why this module exists:
    Phase 6's expected-value math multiplies P(churn) x uplift x LTV. This
    module derives LTV per plan tier from the data instead of guessing it:
    a survival curve per tier, then restricted mean survival time (RMST)
    to a 24-month horizon, times monthly revenue.

Derivation (see notebooks/02_eda.py Section C):
    For each plan tier:
      1. Build a discrete-time survival curve from the snapshot: the
         month-t hazard is the 30-day churn rate of subscribers whose
         tenure is t (life_table_survival below)
      2. Compute RMST(24) = sum of S over the first 24 months
      3. LTV = RMST x monthly_revenue

The constants below are the derivation output as of the current
`data/subscribers.csv`. If the simulator or dataset changes, the
consistency test in tests/test_ltv.py catches drift.

Loading:
    from src.decisions.ltv import LTV_BY_TIER
    # dict: {"Basic": 109, "Standard": 187, "Premium": 296}
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Restricted-mean-survival-time horizon (months). 24 months is standard
# in subscription LTV — captures the full anniversary cycle including
# the m12 churn spike, but doesn't extrapolate too far past the
# observation window (max tenure in the data is 60 months but only ~8%
# of users are past 24 months, so the monthly hazards get noisy past that).
RMST_HORIZON_MONTHS: int = 24


# LTV per plan tier, derived from the survival curves on the current
# dataset via derive_ltv_from_data() below. Rounded to whole dollars.
# See tests/test_ltv.py for the drift-detection test.
LTV_BY_TIER: Dict[str, float] = {
    "Basic":    109.0,   # RMST 12.11 mo x $9/mo   (S(24) = 0.26)
    "Standard": 187.0,   # RMST 13.39 mo x $14/mo  (S(24) = 0.32)
    "Premium":  296.0,   # RMST 15.56 mo x $19/mo  (S(24) = 0.44)
}


# Monthly revenue per tier (kept here so LTV derivation is self-contained;
# also mirrored in src/data/simulate.py::SimConfig.monthly_revenue).
MONTHLY_REVENUE_BY_TIER: Dict[str, float] = {
    "Basic":    9.0,
    "Standard": 14.0,
    "Premium":  19.0,
}


def life_table_survival(durations: np.ndarray,
                        events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete-time survival curve from a cross-sectional snapshot.

    The data is one snapshot: each subscriber is observed once, at their
    current tenure, with a 30-day churn outcome. The users who can churn
    "at month t" are the ones whose tenure IS t right now, so the monthly
    hazard is

        h(t) = churners with tenure t / subscribers with tenure t

    and S(t + 1) = S(t) * (1 - h(t)). This treats the snapshot as a
    synthetic cohort, i.e. it assumes month-t hazards are stable over
    time. (A Kaplan-Meier at-risk set of "tenure >= t" would be wrong
    here: users above t already survived month t, but that survival was
    never observed as an event, so the hazards come out far too small.)

    Args:
        durations: tenure in whole months at the snapshot
        events:    1 if the subscriber churned in the next 30 days, else 0

    Returns:
        (times, survival): S at the END of each month t = 0, 1, ...,
        max tenure. S(0) = 1 is implicit at the start.
    """
    durations = np.asarray(durations).astype(int)
    events = np.asarray(events).astype(int)
    ts, ss = [], []
    s = 1.0
    for t in range(int(durations.max()) + 1):
        at_t = durations == t
        n = int(at_t.sum())
        if n > 0:
            s *= 1 - events[at_t].sum() / n
        ts.append(t + 1)
        ss.append(s)
    return np.array(ts, dtype=float), np.array(ss, dtype=float)


def restricted_mean_survival_time(event_times: np.ndarray,
                                  survival: np.ndarray,
                                  horizon: int = RMST_HORIZON_MONTHS) -> float:
    """Restricted mean survival time up to `horizon`.

    Integrates the survival step function S(t) from 0 to `horizon` using the
    fact that S is constant between event times.
    """
    prev_t, prev_S = 0.0, 1.0
    area = 0.0
    for t_i, S_i in zip(event_times, survival):
        if t_i >= horizon:
            area += prev_S * (horizon - prev_t)
            return area
        area += prev_S * (t_i - prev_t)
        prev_t, prev_S = float(t_i), float(S_i)
    # Loop exited without hitting horizon; extend the last S to horizon
    area += prev_S * (horizon - prev_t)
    return area


def derive_ltv_from_data(df: pd.DataFrame,
                         horizon: int = RMST_HORIZON_MONTHS) -> Dict[str, float]:
    """Compute LTV per plan tier from the survival curves on `df`.

    Args:
        df: subscriber DataFrame with columns `plan_tier`, `tenure_months`,
            `churned_next_30d`.
        horizon: RMST horizon in months.

    Returns:
        {tier: LTV} — LTV in dollars, rounded to the nearest whole dollar
        for stability in downstream comparisons.
    """
    out: Dict[str, float] = {}
    for tier, monthly in MONTHLY_REVENUE_BY_TIER.items():
        sub = df[df["plan_tier"] == tier]
        t, S = life_table_survival(sub["tenure_months"].values,
                                   sub["churned_next_30d"].values)
        rmst = restricted_mean_survival_time(t, S, horizon=horizon)
        out[tier] = round(monthly * rmst)
    return out
