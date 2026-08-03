# Retention Targeting v1 — Decision Memo

**From:** Xi Ru, Data Science
**To:** Marcus Lee (PM, Subscriber Retention), Retention Ops
**Re:** Replacing the blanket m11 credit campaign with cost-aware targeting

---

## TL;DR

The current blanket $5 credit campaign at tenure month 11 is running at an estimated **$4.2k monthly loss** — $7.9k spent to retain $3.8k in expected revenue. The proposed cost-aware targeting policy delivers **+$17.7k net expected value per month** at ~2× the blanket spend but 4× the targeting volume. Net swing: **+$21.9k/month**.

**Recommendation: ship the targeted policy as v1. Sunset the blanket campaign.**

Targeted policy ROI comes in at 1.96×, essentially at the 2.0× threshold the metrics framework set at kickoff. Every scenario in the sensitivity analysis still beats the blanket, and the policy is robust across ±50% uplift assumptions.

---

## Problem

StreamFlix's Retention team currently sends a $5 credit to every subscriber whose tenure hits month 11. The campaign runs monthly, costs ~$8k, and has no measurement of who it actually saves.

Analysis of the last snapshot shows:

- ~1,600 users hit m11 in a given month; all get the credit
- ~5.3% of subscribers churn in any given 30-day window
- Applying our best estimate of intervention uplift (~15% of would-have-churners retained) to the m11 population produces $3.8k in expected retained revenue against $7.9k in cost — a **net loss of $4.2k/month**

The blanket approach fails because most m11 users weren't going to churn anyway. We're paying to retain people who never left.

---

## Recommendation

Replace the blanket campaign with a **cost-aware targeting policy** built on three components:

1. **A calibrated churn model** (XGBoost + Platt calibration) that produces a probability for every subscriber in the base
2. **A per-user expected-value calculation** across a menu of three interventions: curated playlist ($1), $5 credit, Premium upgrade ($12)
3. **A budget-capped allocation rule** that targets users in descending order of expected value until spend is exhausted

For each subscriber, we compute:

```
EV(user, lever) = P(churn | user) × uplift(lever) × LTV(tier) − cost(lever)
```

We target the user only if the best available lever has positive EV.

### Expected impact

| Metric | Current (blanket m11) | Proposed (targeted) | Δ |
|---|---|---|---|
| Users contacted | 1,587 | 6,313 | +4,726 |
| Total cost | $7,935 | $18,404 | +$10,469 |
| Expected retained revenue | $3,765 | $36,128 | **+$32,363** |
| Net expected value | −$4,170 | **+$17,724** | **+$21,894** |
| ROI | 0.47× | **1.96×** | +1.49× |

The targeted policy contacts ~4× more users than the blanket but generates ~10× the retained revenue, because it aims at the users with highest expected value rather than everyone hitting a specific tenure milestone.

### Lever mix under the targeted policy

- `credit_5` (dominant): most users where any lever is positive-EV
- `curated_playlist`: small tail of low-cost interventions
- `premium_upgrade`: rare, gated by a 5% base cap to protect margins

### How the targeting works — lift analysis

The reason cost-aware targeting beats the blanket is simple: the calibrated model ranks users so that the top of the list has a much higher churn concentration than the base rate. Concretely:

| Target | Users contacted (per 10k base) | True churners caught | Share of all churners | **Lift vs random** |
|---|---|---|---|---|
| Top **5%** | 500 | 118 | 22.1% | **4.4×** |
| Top **10%** | 1,000 | 186 | 34.8% | **3.5×** |
| Top **20%** | 2,000 | 276 | 51.7% | **2.6×** |

Reading this: *the top 5% of the base — the 500 users the model flags as highest-risk — contains 22% of all real churners.* If we contacted only those 500 users we'd catch 4.4× more churners than sending 500 random credits.

![Lift chart](figures/04_lift_chart.png)

This is the mechanic that makes the ROI story work — most m11 users the blanket targets aren't going to churn, and the model can tell which few actually might.

---

## Risks & assumptions

**Uplift is a PM assumption, not a measurement.** The 5%/15%/25% uplift figures for the three levers come from the 2024 pilot summary. We ran a sensitivity sweep across ±50%:

- At 50% weaker uplift: ROI 1.70×, still positive, still beats blanket
- At 50% stronger uplift: ROI 1.93×, close to but not clearly above 2.0×
- Peak sensitivity ROI (1.25× uplift): 1.97× — even under favorable assumptions we sit right at the target rather than blowing past it

**The recommendation direction holds across all reasonable uplift assumptions.** But an A/B test per lever in production is the natural next step to nail down real numbers.

**Model discrimination is modest.** PR-AUC = 0.17, ROC-AUC = 0.74, top-10% lift = 3.5× (see the lift chart above — steep decay from 3.5× at the top decile to 1.0× at the bottom, monotonic throughout, which is what a healthy ranking model looks like). The synthetic training data has no unmeasured interactions that a real production dataset would provide. In practice, we'd expect the model to improve as event-stream features (browsing, video-completion rates, notification opens) come online.

**Budget doesn't currently bind.** Only ~13% of subscribers have any positive-EV lever, so the full targeted spend is $18.4k — far below the $200k cap. Adding budget doesn't buy more targeting on this data; we'd need either a stronger model or higher-uplift interventions.

**LTV is derived from the Kaplan-Meier survival curves in Phase 2, not guessed.** Per plan tier, we compute restricted mean survival time (RMST) to a 24-month horizon and multiply by monthly revenue. Current values: **Basic $200, Standard $315, Premium $435** (see `src/decisions/ltv.py`; drift-detection test in `tests/test_ltv.py`). This replaces the earlier ballpark defaults ($72/$140/$228) that assumed strong tier-differentiated churn without checking the data. The KM-derived numbers show more modest tier differentiation because 24-month survival probabilities are all in the 92-96% band — Premium retains longer, but by less than casual intuition suggests. Every downstream number in this memo uses the derived LTV.

---

## Rollout plan

**Week 1–2** — Ship v1 in shadow mode. Score every subscriber daily; log the recommended lever and P(churn) but don't send any interventions. Retention team runs the blanket campaign as usual. Compare recommendations against blanket targets to build stakeholder confidence.

**Week 3** — Turn on the targeted policy for a randomized 50% of the eligible base. Blanket campaign runs on the other 50%. Compare 30-day churn between the two arms.

**Week 5–6** — Read out results. If targeted arm shows lower churn AND lower cost, ramp to 100%.

**Ongoing** — Per-lever A/B tests to measure true uplift. Retrain the model quarterly.

---

## Path to a clean 2.0× ROI

We're at 1.96× — essentially at target but not clearly above it, and sensitivity shows we barely crack it under favorable uplift assumptions. Three levers to move from "at target" to "comfortably above":

1. **Measure real uplift via A/B test.** The 15% credit uplift is a 2024 pilot estimate; if the true value is 18–20%, we'd clear 2.0× cleanly. This is the highest-impact + lowest-effort change and is on the roadmap anyway (see Rollout plan below).
2. **Tier-differentiated offers.** Basic gets cheaper interventions ($1 playlist), Premium gets richer ones ($12 upgrade). Shifts the EV distribution favorably per tier.
3. **Stronger model.** Event-stream features are the biggest bet — playback completion rates, notification response, cross-device usage. Requires engineering investment on the data pipeline side, so this is a Q3 conversation, not a v1 dependency.

**v2 preview (Phase 8, causal uplift model):** On the same experimental subset, a T-learner uplift model delivers 25× more *true* retained revenue than the propensity-based v1 by targeting per-user causal responsiveness rather than absolute risk. That work is queued as a follow-up once we have ~5× the current experimental sample. See [`notebooks/09_policy_comparison.py`](../notebooks/09_policy_comparison.py).

---

## Appendix

- Full methodology: [`notebooks/06_decision_rule.ipynb`](../notebooks/06_decision_rule.ipynb)
- Metric framework (defined at kickoff): [`reports/metrics_framework.md`](./metrics_framework.md)
- Original PM brief: [`reports/scenario_brief.md`](./scenario_brief.md)
- Interactive tool: `streamlit run app/streamlit_app.py`
