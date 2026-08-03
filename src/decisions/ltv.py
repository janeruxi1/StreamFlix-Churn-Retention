"""LTV constants + KM-derived derivation for the Phase 6 decision rule.

Why this module exists:
    Phase 6's expected-value math multiplies P(churn) x uplift x LTV. LTV
    used to be ballpark defaults (Basic $72 = 9 x 8, Standard $140,
    Premium $228) — reasonable ordinal magnitudes but not derived from
    data. This module replaces those defaults with LTVs computed from
    Phase 2's Kaplan-Meier survival curves per plan tier, via restricted
    mean survival time (RMST) to a 24-month horizon.

Derivation (see notebooks/02_eda.py Section H):
    For each plan tier:
      1. Fit a KM survival curve on all subscribers of that tier
         (durations = tenure_months, events = churned_next_30d)
      2. Compute RMST(24) = integral of S(t) from 0 to 24 months
      3. LTV = RMST x monthly_revenue

The constants below are the derivation output as of the current
`data/subscribers.csv`. If the simulator or dataset changes, the
consistency test in tests/test_ltv.py catches drift.

Loading:
    from src.decisions.ltv import LTV_BY_TIER
    # dict: {"Basic": 200, "Standard": 315, "Premium": 435}
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Restricted-mean-survival-time horizon (months). 24 months is standard
# in subscription LTV — captures the full anniversary cycle including
# the m12 churn spike, but doesn't extrapolate too far past the
# observation window (max tenure in the data is 60 months but only ~8%
# of users are past 24 months, so KM estimates get noisy past that).
RMST_HORIZON_MONTHS: int = 24


# LTV per plan tier, derived from KM survival curves on the current
# dataset via derive_ltv_from_data() below. Rounded to whole dollars.
# See tests/test_ltv.py for the drift-detection test.
LTV_BY_TIER: Dict[str, float] = {
    "Basic":    200.0,   # RMST 22.27 mo x $9/mo
    "Standard": 315.0,   # RMST 22.49 mo x $14/mo
    "Premium":  435.0,   # RMST 22.90 mo x $19/mo
}


# Monthly revenue per tier (kept here so LTV derivation is self-contained;
# also mirrored in src/data/simulate.py::SimConfig.monthly_revenue).
MONTHLY_REVENUE_BY_TIER: Dict[str, float] = {
    "Basic":    9.0,
    "Standard": 14.0,
    "Premium":  19.0,
}


def kaplan_meier(durations: np.ndarray,
                 events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Homemade Kaplan-Meier estimator (no lifelines dependency).

    Args:
        durations: tenure at observation for each subject
        events:    1 if the subject churned in the 30-day window, else 0

    Returns:
        (event_times, survival_probs) — the step-function corners of S(t).
        S(0) = 1 implicitly; between event times S is constant.
    """
    durations = np.asarray(durations)
    events = np.asarray(events)
    event_times = np.sort(np.unique(durations[events == 1]))
    survivors = 1.0
    ts, ss = [], []
    for t in event_times:
        n_events_at_t = ((durations == t) & (events == 1)).sum()
        n_at_risk    = (durations >= t).sum()
        if n_at_risk == 0:
            break
        survivors *= (1 - n_events_at_t / n_at_risk)
        ts.append(t)
        ss.append(survivors)
    return np.array(ts, dtype=float), np.array(ss, dtype=float)


def restricted_mean_survival_time(event_times: np.ndarray,
                                  survival: np.ndarray,
                                  horizon: int = RMST_HORIZON_MONTHS) -> float:
    """Restricted mean survival time up to `horizon`.

    Integrates the KM step function S(t) from 0 to `horizon` using the
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
    """Compute LTV per plan tier from the KM curves on `df`.

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
        t, S = kaplan_meier(sub["tenure_months"].values,
                             sub["churned_next_30d"].values)
        rmst = restricted_mean_survival_time(t, S, horizon=horizon)
        out[tier] = round(monthly * rmst)
    return out
