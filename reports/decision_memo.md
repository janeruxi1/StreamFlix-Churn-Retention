# Retention Targeting v1 — Decision Memo

**From:** Xi Ru, Data Science
**To:** Marcus Lee (PM, Subscriber Retention), Retention Ops, Finance
**Re:** Replacing the blanket m11 credit campaign with cost-aware targeting

*All figures are per month for the full ~2.1M-subscriber base, estimated on a 50,000-subscriber sample and scaled up (×42). The data is synthetic; see Limitations.*

---

## TL;DR

**Recommendation: ship v1 (churn-model targeting with a cost-aware decision rule) at the $200k budget, and stop the blanket m11 campaign. Do not ship the v2 uplift model.**

| | Blanket m11 (today) | Targeted v1 (proposed) |
|---|---:|---:|
| Subscribers contacted | 66,654 | 118,986 |
| Spend | $333k | $200k |
| Expected retained revenue | $82k | $374k |
| **Net expected value** | **−$252k** | **+$174k** |
| ROI (revenue / spend) | 0.25× | 1.87× |

- **Swing: +$426k a month** while spending $133k less. The blanket loses money because almost no m11 subscribers were going to churn: v1 would contact only 4% of them.
- **ROI target.** 1.87× is below the 2.0× kickoff target. ROI reaches 2.0× at a budget of about $150k (net EV $155k instead of $174k). Finance should choose between the two; I recommend the full $200k because each extra dollar still returns $1.31 (shadow price 0.31).
- **The $5-credit experiment** confirms the credit works on average (12.6% of would-be churners retained, 95% CI 5–20%, vs the 15% we assumed). It is far too small to learn *who* responds: the uplift model ranks users no better than chance, so v2 is shelved.

---

## Problem

The Retention team sends a $5 credit to everyone reaching month 11, about 67k subscribers a month. Nobody measures who would have stayed anyway. Monthly churn is 5.3%, with spikes at month 2 (end of trial) and month 12 (anniversary).

The PM's ask: for each subscriber, predict 30-day churn, recommend the cheapest worthwhile intervention, and stay within $200k a month.

---

## Part 1 — Churn model

A calibrated `HistGradientBoostingClassifier` (Optuna-tuned, Platt calibration), trained on 30k subscribers and evaluated once on a 10k held-out test set.

| Metric (test set) | Value | Read |
|---|---:|---|
| PR-AUC | **0.177** (bootstrap 95% interval about ±0.03) | 3.3× the 0.053 random baseline |
| ROC-AUC | 0.750 | |
| Top-10% lift | **3.5×** | The riskiest 10% contain 35% of churners |
| Brier | 0.047 | vs 0.051 for always predicting the base rate |

- **Model choice.** Logistic regression (PR-AUC 0.178) and tuned XGBoost (0.178) are statistically tied with the shipped model; every gap is inside the bootstrap interval. HistGBM is kept for practical reasons (sklearn-only, missing values, fast per-user SHAP), and logistic regression is the documented fallback. The tuned variant was chosen before looking at the test set.
- **Calibration by segment** (predicted vs observed churn, test set): within ±13% for every plan tier, engagement cohort and tenure band except **month 12**: predicted 4.9%, observed 8.3% (p = 0.014, n = 291). The model under-predicts the anniversary spike, so v1 under-targets m12 subscribers. Fix before launch: add the anniversary flag to the calibration step, or calibrate by tenure band.
- **Explanations.** SHAP gives each flagged subscriber their top risk drivers (recent viewing, annual billing, logins, days since last login, payment health). These are associations, so the driver-to-intervention map is a set of hypotheses for the per-lever tests, not a causal claim.

---

## Part 2 — Decision rule (v1, what we ship)

```
EV(subscriber, lever) = P(churn) × uplift(lever) × LTV(tier) − cost(lever)
```

**Inputs.**

| Lever | Cost | Uplift (share of would-be churners retained) |
|---|---:|---:|
| `curated_playlist` | $1 | 5% |
| `credit_5` | $5 | 15% (experiment: 12.6%) |
| `premium_upgrade` | $12 | 25% |

**LTV comes from our own retention curves.** The 30-day churn rate at each tenure month gives a monthly hazard; chaining them gives a survival curve per tier, and 24-month restricted mean survival × price gives LTV. 24 months covers the anniversary spike without relying on the thin long-tenure tail (only 9% of subscribers have 24+ months of tenure).

| Tier | Price | Still subscribed at 12 / 24 months | Expected months (of 24) | LTV |
|---|---:|---:|---:|---:|
| Basic | $9 | 44% / 26% | 12.1 | **$109** |
| Standard | $14 | 50% / 32% | 13.4 | **$187** |
| Premium | $19 | 59% / 44% | 15.6 | **$296** |

**The budget binds.** 9.6% of subscribers have a positive-EV lever; covering all of them would cost $415k. With $200k, ranking subscribers by raw EV wastes money on expensive levers. The policy instead charges each dollar its opportunity cost λ (a Lagrange multiplier) and gives each subscriber the lever that maximises `EV − λ × cost`. At $200k, λ = 0.31. This rule earns **33% more** net EV than ranking by raw EV at the same spend ($174k vs $131k).

**Lever mix at $200k.**

| Lever | Subscribers | % contacted | Spend | % spend |
|---|---:|---:|---:|---:|
| `curated_playlist` | 100,800 | 84.7% | $101k | 50.4% |
| `credit_5` | 17,010 | 14.3% | $85k | 42.5% |
| `premium_upgrade` | 1,176 | 1.0% | $14k | 7.1% |

Premium upgrades stay far below the 10k/month ops cap.

**Budget choice.**

| Budget | Net EV | ROI |
|---:|---:|---:|
| $100k | $127k | 2.27× |
| $150k | $155k | 2.04× |
| **$200k** | **$174k** | **1.87×** |
| $250k | $187k | 1.75× |
| $415k (all positive-EV subscribers) | $203k | 1.49× |

**Sensitivity.**
- Uplifts at 0.5× of the assumptions: net EV $18k, ROI 1.32×. At 1.5×: $362k, 2.81×. Targeting beats the blanket throughout; the blanket loses $211k–$292k across the same range.
- LTV horizon of 12 months: net EV $44k. At 36 months: $273k. The direction holds at every horizon.

---

## Part 3 — The $5-credit experiment and the uplift model (v2: not shipped)

A 50/50 randomized holdout tested the $5 credit (one treatment arm, one control, ~25k each).

- **Average effect:** churn 5.28% (control) vs 4.61% (credit), a reduction of **0.67pp (95% CI 0.29–1.05pp)**, i.e. **12.6%** of would-be churners retained (CI 5–20%). That supports the 15% planning figure and should replace it in v1.
- **Who responds:** four uplift learners (S, T, X, class transformation) all score Qini ≈ 0, and their predicted uplift has a rank correlation of +0.02 with each subscriber's true uplift. With a 0.7pp average effect, 25k per arm is enough to confirm the effect exists (minimum detectable effect 0.54pp) but nowhere near enough to learn differences between subscribers.
- **Head-to-head** on 3,010 subscribers that neither model trained on, credit only, $200k budget, scored against the simulator's true effects:

| Policy | Rank correlation with true value | True net value |
|---|---:|---:|
| v1 (churn risk × 15%) | 0.82 | **+$130k** |
| v2 (uplift model) | 0.00 | −$132k |
| Random | −0.01 | −$149k |
| Oracle | 1.00 | +$233k |

v1 captures over half of what perfect targeting would, because the credit's effect scales with churn risk and the churn model estimates risk well. **Keep v1; revisit uplift modelling only with a much larger or repeated experiment.**

---

## Fairness and guardrails

- **Targeting is concentrated by design.** Low-churn groups get few offers: heavy viewers (0.29× the average rate), Basic tier (0.41×), tenure 13–24 months (0.45×), tenure 25+ (0.16×; recall on actual churners 0.16× average). Except for the m12 under-prediction above, these follow lower churn or lower LTV, not miscalibration. Basic's lower targeting reflects its lower LTV ($109), which is a business choice for the PM to confirm.
- **False-positive cost share guardrail (< 30%) is not met: ~72%** of spend goes to subscribers who would not have churned. At a 5% base rate no useful policy can meet this bar, and the EV rule already prices that waste in. I propose replacing it with "net EV > 0 per lever, checked in the holdout".
- **Targeting reach** (≥ 5,000 a month): met (~119k).

---

## Risks

1. **Uplifts for playlist and Premium upgrade are untested.** Playlists are 85% of contacts and half the spend, so they matter most. Test them first.
2. **m12 under-prediction** (above). Fix before launch.
3. **Snapshot assumptions.** LTV treats today's monthly churn rates as stable over time, and scores are from a single snapshot (no out-of-time validation).
4. **Retained-churner value.** LTV is a tier average from month 0; a retained at-risk subscriber may be worth less.

---

## Rollout plan

- **Weeks 1–2: shadow mode.** Score every subscriber daily and log the recommended lever, without sending anything.
- **Week 3: randomized launch.** v1 for a random 50% of the base; blanket for the other 50%.
- **Weeks 5–6: read-out.** Compare 30-day churn and cost. Ramp to 100% if v1 wins.
- **Ongoing:**
  - per-lever holdouts, playlist first;
  - replace the 15% credit uplift with the measured 12.6%;
  - recalibrate by tenure band;
  - retrain quarterly.

---

## Appendix

- Decision rule, budget sweep, sensitivity: [`notebooks/06_decision_rule.ipynb`](../notebooks/06_decision_rule.ipynb)
- Calibration and fairness: [`notebooks/05b_fairness_audit.ipynb`](../notebooks/05b_fairness_audit.ipynb)
- Experiment and uplift models: [`notebooks/08_uplift_modeling.ipynb`](../notebooks/08_uplift_modeling.ipynb), [`notebooks/09_policy_comparison.ipynb`](../notebooks/09_policy_comparison.ipynb)
- Metric framework: [`metrics_framework.md`](./metrics_framework.md) · PM brief: [`scenario_brief.md`](./scenario_brief.md) · Confidence intervals: [`metric_confidence_intervals.md`](./metric_confidence_intervals.md)
- Interactive tool: `streamlit run app/streamlit_app.py`
