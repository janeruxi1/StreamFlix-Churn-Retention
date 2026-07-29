"""Synthetic StreamFlix subscriber dataset generator (v3 -- workplace-grade).

DESIGN PHILOSOPHY
=================
v3 makes every count-style feature multi-window (matching what real
subscription Retention teams track), and replaces simple booleans with
continuous "days since/until" features so the model can learn the right
thresholds itself.

Changes vs v2:
  - support_tickets: 90d → 7d / 30d / 90d (recent tickets are far more
    predictive than older ones)
  - payment_failures: 6mo single count → 30d / 90d / 180d
  - logins_last_30d: NEW activity-count companion to days_since_last_login
  - recent_downgrade (bool) → days_since_plan_change (numeric, -1 = never)
  - promo_expires_soon (bool) → days_until_promo_expires (numeric, -1 = no promo)

Time windows are mathematically nested (7d <= 30d <= 90d) using Poisson
thinning -- the same way real event-stream features are aggregated.

GROUND TRUTH (embedded):
  - Baseline 30-day churn ~5.5%
  - Top drivers: declining engagement trend, recent plan change, expiring
    promo, recent payment failures, recent support tickets, gift_card payment,
    high days_since_login, auto_renew off
  - Stabilizers: tenure, annual billing, multi_profile, Premium tier
  - Tenure spikes at months 2 (trial drop) and 12 (annual reassessment)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimConfig:
    n_subscribers: int = 50_000
    seed: int = 42

    plan_share: dict = field(default_factory=lambda: {
        "Basic": 0.40, "Standard": 0.40, "Premium": 0.20,
    })
    country_share: dict = field(default_factory=lambda: {
        "US": 0.50, "UK": 0.15, "CA": 0.10, "AU": 0.08, "Other": 0.17,
    })
    payment_share: dict = field(default_factory=lambda: {
        "credit_card": 0.65, "paypal": 0.25, "gift_card": 0.10,
    })
    engagement_share: dict = field(default_factory=lambda: {
        "heavy":   0.15, "regular": 0.60, "casual":  0.25,
    })
    monthly_revenue: dict = field(default_factory=lambda: {
        "Basic": 9.0, "Standard": 14.0, "Premium": 19.0,
    })

    base_churn_logit: float = -2.85
    plan_effect: dict = field(default_factory=lambda: {
        "Basic": +0.20, "Standard": 0.0, "Premium": -0.35,
    })
    billing_effect: dict = field(default_factory=lambda: {
        "monthly": +0.20, "annual": -0.50,
    })
    payment_effect: dict = field(default_factory=lambda: {
        "credit_card": 0.0, "paypal": +0.10, "gift_card": +0.55,
    })


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_subscribers(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_subscribers

    # ---- 1. Identity & demographics ----
    subscriber_id = np.arange(2_000_000, 2_000_000 + n)
    plan_tier = rng.choice(list(cfg.plan_share), size=n, p=list(cfg.plan_share.values()))
    country = rng.choice(list(cfg.country_share), size=n, p=list(cfg.country_share.values()))
    payment_method = rng.choice(list(cfg.payment_share), size=n, p=list(cfg.payment_share.values()))

    # ---- 2. Tenure & billing & profile ----
    tenure_months = np.clip(rng.exponential(scale=10.0, size=n), 0, 60).astype(int)

    p_annual = _sigmoid(-1.3 + 0.05 * tenure_months)
    billing_cycle = np.where(rng.random(n) < p_annual, "annual", "monthly")

    p_auto = _sigmoid(
        1.2 + 1.0 * (billing_cycle == "annual").astype(float) + 0.04 * tenure_months
    )
    auto_renew = rng.random(n) < p_auto

    p_multi = _sigmoid(
        -0.7
        + 0.5 * (plan_tier == "Premium").astype(float)
        + 0.3 * (plan_tier == "Standard").astype(float)
        + 0.02 * tenure_months
    )
    multi_profile = rng.random(n) < p_multi

    # ---- 3. Engagement: bimodal cohort + 3 time windows ----
    cohort = rng.choice(
        list(cfg.engagement_share), size=n, p=list(cfg.engagement_share.values()),
    )
    base = np.zeros(n)
    base[cohort == "heavy"]   = rng.gamma(3.0, 12.0, size=(cohort == "heavy").sum())
    base[cohort == "regular"] = rng.gamma(2.5, 6.0,  size=(cohort == "regular").sum())
    base[cohort == "casual"]  = rng.gamma(2.0, 1.5,  size=(cohort == "casual").sum())

    # `base` is the user's typical monthly watch hours.
    # The three time-window columns must be on the same monthly-rate
    # baseline so downstream trend ratios are interpretable:
    #   watch_30d == base when trend == 0
    #   watch_90d == base * 3 when trend == 0  (three months of consistent watching)
    #   watch_7d  == base * 7/30 when trend == 0  (one week's proportion)
    #
    # `trend` modulates short-term behavior. Recent windows feel the
    # trend more strongly than the long window (which averages over more
    # history). The 90d window gets a small inverse adjustment so a
    # user trending UP in the last 7-30 days has slightly LOWER 90d
    # totals (because their longer-run rate was lower), and vice versa.
    trend = rng.normal(loc=0.0, scale=0.30, size=n)
    watch_hours_last_90d = (base * 3 * np.exp(-trend * 0.2)).round(1)
    watch_hours_last_30d = (base * np.exp(trend * 0.5)).round(1)
    watch_hours_last_7d  = (base * (7 / 30) * np.exp(trend * 1.5)).round(2)

    def _distinct_from_hours(hours: np.ndarray, sd_noise: float) -> np.ndarray:
        return np.clip(
            np.round(hours * 0.7 + rng.normal(0, sd_noise, size=n)),
            0, None,
        ).astype(int)

    distinct_titles_7d  = _distinct_from_hours(watch_hours_last_7d,  1.0)
    distinct_titles_30d = _distinct_from_hours(watch_hours_last_30d, 1.5)
    distinct_titles_90d = _distinct_from_hours(watch_hours_last_90d, 2.0)

    # Days since last login + activity-count companion (logins_last_30d)
    base_dsl = rng.exponential(scale=4.0, size=n)
    disengaged = rng.random(n) < 0.10
    days_since_last_login = np.clip(
        base_dsl + disengaged * rng.uniform(7, 30, size=n), 0, 60,
    ).astype(int)

    # Login density (count) — complements recency
    login_rate = np.zeros(n)
    login_rate[cohort == "heavy"]   = 26
    login_rate[cohort == "regular"] = 15
    login_rate[cohort == "casual"]  = 5
    logins_last_30d = rng.poisson(np.where(disengaged, login_rate / 3, login_rate)).astype(int)

    # ---- 4. Support tickets: 7d / 30d / 90d via Poisson thinning ----
    # Sample a per-user 90-day base rate, then split into nested windows.
    base_ticket_rate_90d = 0.22 + 0.10 * disengaged.astype(float)
    tickets_0_7   = rng.poisson(base_ticket_rate_90d * (7 / 90))
    tickets_8_30  = rng.poisson(base_ticket_rate_90d * (23 / 90))
    tickets_31_90 = rng.poisson(base_ticket_rate_90d * (60 / 90))
    support_tickets_7d  = tickets_0_7
    support_tickets_30d = tickets_0_7 + tickets_8_30
    support_tickets_90d = tickets_0_7 + tickets_8_30 + tickets_31_90

    # ---- 5. Payment failures: 30d / 90d / 180d via Poisson thinning ----
    base_fail_rate_180d = (
        0.10
        + 0.20 * (payment_method == "gift_card").astype(float)
        + 0.08 * (tenure_months < 3).astype(float)
    )
    fails_0_30    = rng.poisson(base_fail_rate_180d * (30 / 180))
    fails_31_90   = rng.poisson(base_fail_rate_180d * (60 / 180))
    fails_91_180  = rng.poisson(base_fail_rate_180d * (90 / 180))
    payment_failures_30d  = fails_0_30
    payment_failures_90d  = fails_0_30 + fails_31_90
    payment_failures_180d = fails_0_30 + fails_31_90 + fails_91_180

    # ---- 6. Lifecycle: days_since_plan_change, days_until_promo_expires ----
    # ~7% had a recent (0-90d) plan change; ~8% had an older (>90d) one
    plan_change_recent = rng.random(n) < 0.07
    plan_change_old = (rng.random(n) < 0.08) & (~plan_change_recent)
    days_since_plan_change = np.full(n, -1, dtype=int)
    days_since_plan_change[plan_change_recent] = rng.integers(
        0, 90, size=plan_change_recent.sum(),
    )
    days_since_plan_change[plan_change_old] = rng.integers(
        91, 366, size=plan_change_old.sum(),
    )

    # ~12% have an active promo (1-90 days remaining)
    promo_active = rng.random(n) < 0.12
    days_until_promo_expires = np.full(n, -1, dtype=int)
    days_until_promo_expires[promo_active] = rng.integers(
        1, 91, size=promo_active.sum(),
    )

    # ---- 7. Monthly revenue ----
    monthly_revenue = np.array([cfg.monthly_revenue[t] for t in plan_tier])

    # ---- 8. Churn outcome -- the embedded ground truth ----
    churn_logit = np.full(n, cfg.base_churn_logit, dtype=float)
    churn_logit += np.array([cfg.plan_effect[t] for t in plan_tier])
    churn_logit += np.array([cfg.billing_effect[b] for b in billing_cycle])
    churn_logit += np.array([cfg.payment_effect[p] for p in payment_method])
    churn_logit += np.where(auto_renew, 0.0, +0.65)
    churn_logit += np.where(multi_profile, -0.35, 0.0)

    # Tenure
    churn_logit += np.clip(-0.02 * tenure_months, -0.6, 0.0)
    churn_logit += np.where(tenure_months == 2, +0.55, 0.0)
    churn_logit += np.where((tenure_months >= 11) & (tenure_months <= 12), +0.45, 0.0)

    # Engagement (strongest signal)
    churn_logit += -1.4 * trend
    churn_logit += -0.04 * np.minimum(watch_hours_last_30d, 25)
    churn_logit += 0.06 * np.maximum(days_since_last_login - 7, 0)
    churn_logit += -0.015 * np.minimum(logins_last_30d, 30)
    churn_logit += np.where(cohort == "casual", +0.25, 0.0)

    # Recent vs older support tickets weighted differently
    churn_logit += 0.55 * support_tickets_7d
    churn_logit += 0.20 * (support_tickets_30d - support_tickets_7d)
    churn_logit += 0.05 * (support_tickets_90d - support_tickets_30d)

    # Recent vs older payment failures weighted differently
    churn_logit += 0.70 * payment_failures_30d
    churn_logit += 0.30 * (payment_failures_90d - payment_failures_30d)
    churn_logit += 0.10 * (payment_failures_180d - payment_failures_90d)

    # Plan change recency: peaks at day 0, fades over 90 days
    has_recent_change = (days_since_plan_change >= 0) & (days_since_plan_change <= 90)
    churn_logit += np.where(
        has_recent_change,
        +0.60 - 0.006 * np.clip(days_since_plan_change, 0, 90),
        0.0,
    )

    # Expiring promo: peaks at day 1, fades by day 14
    has_active_promo = days_until_promo_expires >= 0
    churn_logit += np.where(
        has_active_promo & (days_until_promo_expires <= 14),
        +0.55 - 0.035 * np.clip(days_until_promo_expires, 0, 14),
        0.0,
    )

    churn_prob = _sigmoid(churn_logit)
    churned_next_30d = rng.binomial(1, churn_prob).astype(int)

    # ---- 8.5. Treatment assignment & counterfactual outcomes (uplift) ----
    # Simulates a randomized retention experiment for Phase 8 uplift
    # modeling. 50% of users receive a lever from the intervention menu;
    # the other 50% are control. For each treated user we draw a
    # counterfactual outcome under the assigned lever with per-user
    # heterogeneity + a small sleeping-dogs segment.
    #
    # Adds columns: treated, treatment_lever, churned_if_treated,
    # y_observed, true_uplift.
    # Backward compatible: churned_next_30d unchanged (still the pure
    # control potential outcome that Phase 4-6 train and score against).

    # Base uplift (reduction in P(churn)) per lever, before heterogeneity
    _LEVER_BASE_UPLIFT = {
        "email_nudge":     0.03,
        "credit_5":        0.10,
        "credit_10":       0.15,
        "content_push":    0.06,
        "premium_upgrade": 0.18,
    }
    _LEVER_NAMES = np.array(list(_LEVER_BASE_UPLIFT.keys()))

    # 50/50 treatment assignment
    treated = (rng.random(n) < 0.5).astype(int)

    # ~5% sleeping dogs: treatment INCREASES their churn (they were about
    # to renew but the "we miss you" nudge reminds them to cancel).
    is_sleeping_dog = rng.random(n) < 0.05

    # Uniform lever assignment for treated; "none" for control
    lever_idx = rng.integers(0, len(_LEVER_NAMES), size=n)
    treatment_lever = np.where(treated == 1, _LEVER_NAMES[lever_idx], "none")

    # Base uplift for the assigned lever
    base_uplift = np.array(
        [_LEVER_BASE_UPLIFT.get(l, 0.0) for l in treatment_lever]
    )

    # Segment-based heterogeneity modulators (multiplicative)
    mod = np.ones(n)
    # Users with recent payment failures respond STRONGLY to credit levers
    credit_mask = np.isin(treatment_lever, ["credit_5", "credit_10"])
    mod[credit_mask & (payment_failures_30d > 0)] *= 1.6
    # Casual users respond well to content pushes (haven't found things to watch)
    mod[(treatment_lever == "content_push") & (cohort == "casual")] *= 1.5
    # Heavy users don't need email nudges (already engaged)
    mod[(treatment_lever == "email_nudge") & (cohort == "heavy")] *= 0.3
    # Trial-drop window (tenure=2) responds very well to a small credit
    mod[(treatment_lever == "credit_5") & (tenure_months == 2)] *= 1.8
    # Users with a soon-to-expire promo respond well to a bigger credit
    mod[(treatment_lever == "credit_10")
        & (days_until_promo_expires >= 0)
        & (days_until_promo_expires <= 14)] *= 1.5

    # Sleeping dogs: any treatment BACKFIRES (negative uplift)
    mod[is_sleeping_dog & (treated == 1)] = -0.6

    # True per-user uplift for the assigned lever
    # Convention: positive = churn REDUCTION (retention lift)
    true_uplift = np.where(treated == 1, base_uplift * mod, 0.0)

    # Counterfactual outcome: p_treated = clip(p_control - true_uplift)
    p_treated = np.clip(churn_prob - true_uplift, 0.001, 0.999)
    churned_if_treated = rng.binomial(1, p_treated).astype(int)

    # Observed outcome under the actual assignment (what the retention
    # team would see): control users see churned_next_30d, treated users
    # see churned_if_treated.
    y_observed = np.where(treated == 1, churned_if_treated, churned_next_30d)

    # ---- 9. Assemble ----
    df = pd.DataFrame({
        "subscriber_id": subscriber_id,
        "tenure_months": tenure_months,
        "plan_tier": plan_tier,
        "billing_cycle": billing_cycle,
        "country": country,
        "payment_method": payment_method,
        "auto_renew": auto_renew,
        "multi_profile": multi_profile,
        "engagement_cohort": cohort,
        "watch_hours_last_7d": watch_hours_last_7d,
        "watch_hours_last_30d": watch_hours_last_30d,
        "watch_hours_last_90d": watch_hours_last_90d,
        "distinct_titles_7d": distinct_titles_7d,
        "distinct_titles_30d": distinct_titles_30d,
        "distinct_titles_90d": distinct_titles_90d,
        "days_since_last_login": days_since_last_login,
        "logins_last_30d": logins_last_30d,
        "support_tickets_7d": support_tickets_7d,
        "support_tickets_30d": support_tickets_30d,
        "support_tickets_90d": support_tickets_90d,
        "payment_failures_30d": payment_failures_30d,
        "payment_failures_90d": payment_failures_90d,
        "payment_failures_180d": payment_failures_180d,
        "days_since_plan_change": days_since_plan_change,
        "promo_active": promo_active,
        "days_until_promo_expires": days_until_promo_expires,
        "monthly_revenue": monthly_revenue,
        "churned_next_30d": churned_next_30d,
        # Phase 8: uplift-model experimental columns
        "treated": treated,
        "treatment_lever": treatment_lever,
        "churned_if_treated": churned_if_treated,
        "y_observed": y_observed,
        "true_uplift": true_uplift.round(4),
    })

    return df


# Columns added in v4 for the Phase 8 uplift experiment.
# The pre-experiment baseline file (data/subscribers.csv) DROPS these so
# Phase 4-6 read the same data a real Retention team would have had before
# they ran the A/B holdout. The experiment file keeps them.
_EXPERIMENT_COLS = [
    "treated", "treatment_lever", "churned_if_treated",
    "y_observed", "true_uplift",
]


def main(
    baseline_path: str | Path = "data/subscribers.csv",
    experiment_path: str | Path = "data/subscribers_experiment.csv",
) -> None:
    df = simulate_subscribers()

    # v3 baseline snapshot -- pre-experiment production data.
    # This is the file Phase 4 (churn model), Phase 4b (model bake-off),
    # Phase 5 (SHAP), and Phase 6 (blanket-vs-targeted policy) all use.
    # It represents the world before any randomized retention experiment
    # was run -- everyone's outcome is their untreated potential outcome.
    baseline_out = Path(baseline_path)
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    df_baseline = df.drop(columns=[c for c in _EXPERIMENT_COLS if c in df.columns])
    df_baseline.to_csv(baseline_out, index=False)
    print(f"Wrote v3 BASELINE  ({len(df):,} rows x {df_baseline.shape[1]} cols) "
          f"-> {baseline_out}")

    # v4 experiment snapshot -- adds the randomized treatment layer.
    # This is what a Retention team would have AFTER running an A/B
    # holdout. Phase 8 (uplift modeling) consumes this file.
    experiment_out = Path(experiment_path)
    df.to_csv(experiment_out, index=False)
    print(f"Wrote v4 EXPERIMENT ({len(df):,} rows x {df.shape[1]} cols) "
          f"-> {experiment_out}\n")

    rate = df["churned_next_30d"].mean()
    print(f"Overall 30-day churn rate: {rate:.2%}  "
          f"({'OK in band' if 0.04 <= rate <= 0.08 else 'OUT OF BAND'})\n")

    print("Engagement cohort distribution:")
    print(df["engagement_cohort"].value_counts(normalize=True).round(3))

    heavy = (df["watch_hours_last_30d"] > 30).mean()
    casual = (df["watch_hours_last_30d"] < 3).mean()
    print(f"\nHeavy users (>30 hrs/mo):  {heavy:.1%}  (target: 10-15%)")
    print(f"Casual users (<3 hrs/mo):  {casual:.1%}  (target: 20-30%)")

    print(f"\nMean tickets per user (90d): {df['support_tickets_90d'].mean():.3f}")
    print(f"Mean tickets per user (7d):  {df['support_tickets_7d'].mean():.3f}")
    print(f"7d should be ~ 90d / 13 (proportional time)")

    print(f"\nMean payment failures per user (180d): {df['payment_failures_180d'].mean():.3f}")
    print(f"Mean payment failures per user (30d):  {df['payment_failures_30d'].mean():.3f}")

    print(f"\nUsers with a plan change in last 90d: "
          f"{((df['days_since_plan_change'] >= 0) & (df['days_since_plan_change'] <= 90)).mean():.1%}")
    print(f"Users with active promo: {df['promo_active'].mean():.1%}")

    # Treatment experiment stats (Phase 8)
    print(f"\nTreatment/control (Phase 8 uplift experiment):")
    print(f"  Treated:   {df['treated'].mean():.1%}")
    print(f"  Control:   {(1 - df['treated']).mean():.1%}")
    print(f"  Lever mix (treated only):")
    for lever, count in df[df["treated"] == 1]["treatment_lever"].value_counts().items():
        print(f"    {lever:<18}  n={count:>5,}  ({count / (df['treated'] == 1).sum():.1%})")
    print(f"  Mean true uplift (treated): "
          f"{df.loc[df['treated'] == 1, 'true_uplift'].mean():.4f}")
    print(f"  Sleeping dogs (~5%, negative uplift): "
          f"{(df.loc[df['treated'] == 1, 'true_uplift'] < 0).mean():.1%}")
    control_churn = df.loc[df["treated"] == 0, "y_observed"].mean()
    treated_churn = df.loc[df["treated"] == 1, "y_observed"].mean()
    print(f"  Naive ATE (control - treated churn rate): "
          f"{control_churn - treated_churn:+.4f}")

    print("\nUnivariate churn correlations (>= 0.15 = strong):")
    for col in [
        "watch_hours_last_7d", "watch_hours_last_30d", "watch_hours_last_90d",
        "support_tickets_30d", "payment_failures_30d",
        "days_since_last_login", "logins_last_30d",
    ]:
        corr = df[col].corr(df["churned_next_30d"])
        flag = "  **" if abs(corr) > 0.15 else ""
        print(f"  {col:<30}  {corr:+.3f}{flag}")


if __name__ == "__main__":
    main()
