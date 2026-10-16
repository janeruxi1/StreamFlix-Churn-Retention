# 📝 PM Brief — Subscriber Retention Program

**From:** Marcus Lee, Sr. PM, Subscriber Retention
**To:** Data Science Team
**Re:** Cost-aware churn-targeting model for the monthly paid tier

---

## Background

StreamFlix's paid subscription business has ~2.1M active monthly subscribers across three tiers (Basic, Standard, Premium). **Monthly churn averages ~5.5%** but spikes meaningfully at the 2-month and 12-month tenure points (free-trial-to-paid drop-off and annual-anniversary reassessment).

Up to now, the Retention team has run a single broad-brush campaign — a generic $5 monthly credit offered to *any* subscriber who hits month 11 (roughly 65–70k people a month) — which costs ~$330k/month and has unclear ROI because we never measure who would have stayed anyway.

The team wants to **move from blanket campaigns to a cost-aware, model-driven targeting system.**

## Business problem

> **For each subscriber, predict the probability they will churn in the next 30 days and recommend the cheapest intervention (if any) expected to retain them — within a fixed monthly budget — to maximize net dollar value to the business.**

Three decisions need to be made for each subscriber every month:
1. Are they at meaningful risk of churning?
2. If yes, which intervention should we offer?
3. What's the expected ROI if we do?

## Intervention menu

The Retention team has negotiated three pre-built interventions with the relevant ops teams:

| Intervention | Cost per offer | Estimated uplift: share of would-be churners retained | Notes |
|---|---|---|---|
| **Curated playlist email** | $1 | 5% | Lightweight; safe to offer anyone |
| **$5 monthly credit** | $5 | 15% | The current default; assume real cost = $5 of LTV |
| **1-month free Premium upgrade** | $12 | 25% | Best uplift but tier-upgrade pressure on revenue |

(These estimates come from an earlier pilot. Treat them as planning assumptions, not measured truth: a randomized holdout for the $5 credit is running and will report back.)

## Economic parameters

| Parameter | Value | Notes |
|---|---|---|
| Monthly price | Basic $9, Standard $14, Premium $19 | LTV per churn averted should be estimated from our own retention data, not assumed |
| Monthly retention budget | **$200,000** | Hard cap, whole subscriber base |
| Decision horizon | 30 days | Predict churn in next 30 days |

## Decision sought

> **Recommend a targeting policy** that maximizes expected net retention value subject to the $200k monthly budget. The policy must specify:
>
> 1. A risk-probability threshold above which we engage a subscriber
> 2. A rule for assigning the right intervention to each engaged subscriber (cheapest one expected to retain them)
> 3. Expected number of churns averted and net dollar value at the chosen threshold

## Stakeholders

- **Primary decision-maker:** Marcus Lee (PM, Retention)
- **Finance:** must sign off on expected ROI
- **Ops:** capacity-constrained on the Premium upgrade — cap at 10k offers/month across the base
- **CX team:** wants to ensure we're not over-targeting one-time grievance churners

## What we want from the data science team

1. A **calibrated churn-probability model** for the next 30 days
2. **Explainability** — for each at-risk subscriber, why the model flagged them, in PM language
3. A **cost-aware decision rule** that picks the optimal intervention per subscriber
4. A **decision memo** with the recommended policy, expected savings, and sensitivity to budget/cost assumptions
5. A **Streamlit decision tool** the Retention team can use to score new subscriber profiles ad hoc

## Risks / open questions

- **Calibration matters more than discrimination.** A model that ranks well but is miscalibrated will mis-size the intervention budget. Calibration is a first-class metric.
- **Segments may need different policies.** New users (tenure < 3 months) may respond very differently to a curated playlist than tenured users.
- **Anchoring.** The curated-playlist arm of the pilot had a small treatment group (~3k); its 5% estimate carries wider uncertainty than the other two.
