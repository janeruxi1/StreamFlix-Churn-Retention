"""Data loading utilities for the StreamFlix subscriber dataset.

Two flavors are produced by src/data/simulate.py:

    data/subscribers.csv             -- v3 BASELINE (28 cols). Pre-experiment
                                        production data. Used by Phase 4
                                        (churn model), Phase 4b (bake-off),
                                        Phase 5 (SHAP), Phase 6 (blanket vs
                                        targeted policy), Phase 7 (memo/app).
    data/subscribers_experiment.csv  -- v4 EXPERIMENT (33 cols). Adds the
                                        randomized treatment layer that
                                        Phase 8 (uplift modeling) needs.

The split mirrors the project's real story: build a propensity-based
targeting policy against the blanket-campaign baseline first, THEN run an
A/B holdout and learn per-user uplift as the second-order improvement.
"""
from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = {
    # Identity & demographics
    "subscriber_id", "tenure_months", "plan_tier", "billing_cycle",
    "country", "payment_method",
    # Account state
    "auto_renew", "multi_profile", "promo_active",
    # Engagement
    "engagement_cohort",
    "watch_hours_last_7d", "watch_hours_last_30d", "watch_hours_last_90d",
    "distinct_titles_7d", "distinct_titles_30d", "distinct_titles_90d",
    "days_since_last_login", "logins_last_30d",
    # Support
    "support_tickets_7d", "support_tickets_30d", "support_tickets_90d",
    # Billing health
    "payment_failures_30d", "payment_failures_90d", "payment_failures_180d",
    # Lifecycle
    "days_since_plan_change", "days_until_promo_expires",
    # Economics & target
    "monthly_revenue", "churned_next_30d",
}

# Experiment file adds these five columns on top of EXPECTED_COLUMNS.
EXPERIMENT_COLUMNS = {
    "treated", "treatment_lever", "churned_if_treated",
    "y_observed", "true_uplift",
}


def load_subscribers(path: str | Path = "data/subscribers.csv") -> pd.DataFrame:
    """Load the v3 baseline subscriber dataset (pre-experiment).

    This is the file Phase 4-6 consume: it represents the world before any
    randomized retention experiment was run, so `churned_next_30d` is the
    only outcome column and it means "would this user churn under no
    intervention?"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python src/data/simulate.py"
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def load_subscribers_experiment(
    path: str | Path = "data/subscribers_experiment.csv",
) -> pd.DataFrame:
    """Load the v4 experimental subscriber dataset (post A/B holdout).

    Same rows as the baseline file, plus five columns from a randomized
    50/50 treatment experiment: `treated`, `treatment_lever`,
    `churned_if_treated`, `y_observed`, `true_uplift`. Phase 8 uses this
    to train uplift models; Phase 4-6 should NOT use this file so their
    narrative stays "here's what we could do without an experiment."
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python src/data/simulate.py"
        )
    df = pd.read_csv(path)
    missing = (EXPECTED_COLUMNS | EXPERIMENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in experiment file: {missing}. "
            f"Regenerate with: python src/data/simulate.py"
        )
    return df
