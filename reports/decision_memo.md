# Retention Targeting v1 — Decision Memo

**From:** Xi Ru, Data Science
**To:** Marcus Lee (PM, Subscriber Retention), Retention Ops
**Re:** Replacing the blanket m11 credit campaign with cost-aware targeting

---

## TL;DR

The current blanket $5 credit campaign at tenure month 11 is running at an estimated **$6.3k monthly loss** — $7.9k spent to retain $1.6k in expected revenue. The proposed cost-aware targeting policy delivers **+$3.3k net expected value per month** at less than 6% of the current spend. Net swing: **+$9.6k/month**.

**Recommendation: ship the targeted policy as v1. Sunset the blanket campaign.**

Targeted policy ROI comes in at 1.72×, below the 2.0× threshold the metrics framework set at kickoff. Every scenario in the sensitivity analysis still beats the blanket, so this is worth shipping — but closing the gap to 2.0× is the natural v1.1 iteration.

---

## Problem

StreamFlix's Retention team currently sends a $5 credit to every subscriber whose tenure hits month 11. The campaign runs monthly, costs ~$8k, and has no measurement of who it actually saves.

Analysis of the last snapshot shows:

- ~1,600 users hit m11 in a given month; all get the credit
- ~5.3% of subscribers churn in any given 30-day window
- Applying our best estimate of intervention uplift (~15% of would-have-churners retained) to the m11 population produces $1.6k in expected retained revenue against $7.9k in cost — a **net loss of $6.3k/month**

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
| Users contacted | 1,587 | 1,510 | −77 |
| Total cost | $7,935 | $4,557 | **−$3,378** |
| Expected retained revenue | $1,606 | $7,824 | **+$6,218** |
| Net expected value | −$6,329 | **+$3,267** | **+$9,596** |
| ROI | 0.20× | **1.72×** | +1.52× |

### Lever mix under the targeted policy

- `credit_5` (dominant): most users where any lever is positive-EV
- `curated_playlist`: small tail of low-cost interventions
- `premium_upgrade`: rare, gated by a 5% base cap to protect margins

---

## Risks & assumptions

**Uplift is a PM assumption, not a measurement.** The 5%/15%/25% uplift figures for the three levers come from the 2024 pilot summary. We ran a sensitivity sweep across ±50%:

- At 50% weaker uplift: ROI 1.48×, still positive, still beats blanket
- At 50% stronger uplift: ROI 1.87×, closer to but still below 2.0× target

**The recommendation direction holds across all reasonable uplift assumptions.** But an A/B test per lever in production is the natural next step to nail down real numbers.

**Model discrimination is modest.** PR-AUC = 0.17 (~3.3× lift over random), ROC-AUC = 0.74. The synthetic training data has no unmeasured interactions that a real production dataset would provide. In practice, we'd expect the model to improve as event-stream features (browsing, video-completion rates, notification opens) come online.

**Budget doesn't currently bind.** Only ~3% of subscribers have any positive-EV lever, so the full targeted spend is $4.5k — far below the $200k cap. Adding budget doesn't buy more targeting on this data; we'd need either a stronger model or higher-uplift interventions.

**LTV assumptions.** Basic $72 (9 × 8), Standard $140 (14 × 10), Premium $228 (19 × 12). These follow the tier retention curves in the scenario brief. If they're materially off, EV rankings could shift — the sensitivity check on uplift covers directional risk but not this axis.

---

## Rollout plan

**Week 1–2** — Ship v1 in shadow mode. Score every subscriber daily; log the recommended lever and P(churn) but don't send any interventions. Retention team runs the blanket campaign as usual. Compare recommendations against blanket targets to build stakeholder confidence.

**Week 3** — Turn on the targeted policy for a randomized 50% of the eligible base. Blanket campaign runs on the other 50%. Compare 30-day churn between the two arms.

**Week 5–6** — Read out results. If targeted arm shows lower churn AND lower cost, ramp to 100%.

**Ongoing** — Per-lever A/B tests to measure true uplift. Retrain the model quarterly.

---

## Path to 2.0× ROI

Three levers to close the gap, ordered by effort:

1. **Renegotiate the intervention menu.** If the true uplift on `credit_5` is 20% instead of 15% (well within the sensitivity range), ROI clears 2.0× at current model discrimination. A real A/B test resolves this.
2. **Tier-differentiated offers.** Basic gets cheaper interventions ($1 playlist), Premium gets richer ones ($12 upgrade). Shifts the EV distribution favorably per tier.
3. **Stronger model.** Event-stream features are the biggest bet — playback completion rates, notification response, cross-device usage. Requires engineering investment on the data pipeline side, so this is a Q3 conversation, not a v1 dependency.

---

## Appendix

- Full methodology: [`notebooks/06_decision_rule.ipynb`](../notebooks/06_decision_rule.ipynb)
- Metric framework (defined at kickoff): [`reports/metrics_framework.md`](./metrics_framework.md)
- Original PM brief: [`reports/scenario_brief.md`](./scenario_brief.md)
- Interactive tool: `streamlit run app/streamlit_app.py`
