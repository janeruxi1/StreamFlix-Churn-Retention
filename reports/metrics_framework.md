# 📊 Metric Framework

Defined before any modeling, by decision-relevance rather than statistical convenience.

---

## Primary metric — drives the ship decision

| Metric | What it measures | Target |
|---|---|---|
| **Expected program ROI** | (Expected $ revenue saved by averted churns − total intervention cost) / total intervention cost | **≥ 2.0×** (ship bar) |

ROI is the only metric the PM actually cares about. Everything else exists to make this number credible.

---

## Secondary metrics — model quality

| Metric | Why it matters |
|---|---|
| **PR-AUC** | Primary discrimination metric. Churn is imbalanced (~5.5% base rate), so PR is more informative than ROC. |
| **ROC-AUC** | Sanity check; widely understood. |
| **Brier score + calibration-in-the-large** | Brier mixes discrimination and calibration and scales with the base rate; predicted vs observed churn (overall and by segment) is the direct calibration check. Without good calibration, the EV rule mis-sizes the budget. |
| **Calibration curve (reliability)** | Visual diagnostic; reviewers expect to see it. |
| **Recall @ top-K decile** | How many of the *actually-churning* users are we capturing in our targeted group? |
| **Precision @ top-K decile** | How wasted is the budget? |

---

## Guardrails — must not regress

| Guardrail | Threshold | Why |
|---|---|---|
| **False-positive cost share** | < 30% of total intervention cost wasted on non-churners | We're going to over-target some non-churners (that's fine); but if the majority of budget goes to people who wouldn't have churned, the program is wasteful. *(Revisited in the decision memo: at a ~5% base rate this bar can't be met by any useful policy, and the EV rule already prices the waste in.)* |
| **Premium-upgrade offer volume** | ≤ 10,000 / month | Ops capacity cap. |
| **Targeting reach** | ≥ 5,000 unique subscribers / month | Below this volume, the program isn't worth the operational overhead. |

---

## Operational metrics — track post-launch

After shipping a policy, we'd track these in a dashboard. Not blocking the ship decision.

- Actual realized churn rate among targeted vs un-targeted (lift validation)
- Intervention acceptance rate by offer type
- Time-to-churn distribution for retained users
- Support ticket volume from the targeted cohort

---

## What we are deliberately NOT optimizing

Stating these explicitly avoids scope creep.

- **Lifetime value prediction beyond a 30-day horizon.** We only need the 30-day churn probability + LTV-per-churn-averted plugged in.
- **Causal uplift modeling** for v1. The intervention uplift estimates are planning inputs from a pilot. Learning user-specific treatment effects needs randomized data and is a separate, harder problem; Phase 8 tries it on the $5-credit holdout once that data exists.
- **Multi-month policy** (e.g., "save this user for the next 6 months"). Monthly cadence; monthly decision.

---

## Decision rule (preview — full math in `notebooks/06_decision_rule`)

For each subscriber, for each intervention `t` ∈ {none, playlist, credit, upgrade}:

```
expected_value(subscriber, t) = P(churn | subscriber) × LTV(subscriber) × uplift(t) − cost(t)
```

Pick `t* = argmax expected_value`.
Engage the subscriber only if `expected_value(subscriber, t*) > 0`.
Subject to: total cost across all engaged subscribers ≤ $200k. When that budget binds, each dollar is charged its opportunity cost (shadow price) λ, i.e. pick `argmax expected_value − λ × cost`.

The cost-aware decision rule isn't a model thing; it's a *policy* thing. The model just produces the calibrated `P(churn | subscriber)`.
