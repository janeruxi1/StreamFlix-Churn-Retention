# Project Summary — StreamFlix Churn Retention

**What was built + what was found.** Single entry-point catalog for reviewers, hiring managers, and future-me. Every number here is verified against the notebook output (see the `verify_memo.py`-style sanity check in `reports/decision_memo.md`'s "verification pass" audit).

---

## 60-second overview

StreamFlix's retention team runs a v0 blanket $5-credit campaign to every user at tenure month 11, losing **$4.8k/month**. This project delivers a cost-aware retention tool in three layers, each measured against that same v0 baseline:

- **Part 1 — Churn model:** predict P(churn) per user
- **Part 2 — Decision rule (v1):** turn P(churn) + LTV + assumed uplift → EV rule → ship
- **Part 3 — Uplift enhancement (v2):** replace assumed uplift with per-user learned causal uplift for `credit_5`

**Headline outcome.** Ship v1 now for a **+$23.9k/month swing at 1.64× ROI**. v2 is queued — it would deliver **~21.9× more true retained revenue than blanket** on ground truth, pending a production A/B holdout for validation.

---

## Headline numbers at a glance

| Metric | v0 (blanket) | v1 (decision rule) | v2 (uplift) |
|---|---:|---:|---:|
| **Users targeted** | 1,587 | 10,933 | ~15k on credit_5-treated ground truth |
| **Total cost** | $7,935 | $30,162 | ~$75k on credit_5-treated ground truth |
| **Net EV / month (model-estimated)** | −$4,798 | **+$19,087** | not directly comparable — see Part 3 |
| **ROI** | 0.40× | 1.64× (at $200k ceiling) | not directly comparable |
| **True retained revenue (ground truth, credit_5-treated subset)** | $21,180 | $83,859 | **$463,242** |
| **% of oracle ceiling captured** | ~3% | ~11% | **~62%** |

Two views because Part 2 and Part 3 measure different things:
- **Part 2 numbers** = model's self-reported EV across the full 50k-user base (v0 vs v1 comparison).
- **Part 3 numbers** = ground-truth retained revenue on the 25k credit_5-treated subset (v0 vs v2 comparison).

---

## Part 1: The churn model

### What was built

- **Feature engineering** (`03_feature_engineering.py`) — 17 engineered features across engagement / recency / lifecycle / composite groups
- **Family bake-off** (`04b_model_comparison.py`) — 4-family comparison: LR baseline (with LogisticRegressionCV), XGBoost (Optuna-tuned), HistGradientBoosting (Optuna-tuned), Random Forest
- **Production training** (`04_modeling.py`) — train default HistGBM + Optuna-tuned HistGBM, pick winner on held-out **TEST** PR-AUC, Platt-calibrate the winner, persist via `src/models/production.py`
- **Interpretability** (`05_shap_levers.py`) — SHAP on the uncalibrated HistGBM with two-layer diagnostic → tactical crosswalk (35 features mapped to 3 operational levers via `src/models/explain.py:FEATURE_INTERVENTION_MAP`)
- **Fairness audit** (`05b_fairness_audit.py`) — PR-AUC, Brier, calibration curves, targeting-rate parity, recall parity across plan_tier / engagement_cohort / tenure_bucket / country, with auto-flag concerns loop

### Key findings

| Finding | Number / evidence |
|---|---|
| PR-AUC (primary metric for ~5% positive class) | **0.20** |
| ROC-AUC | 0.77 |
| Brier score (calibration) | 0.047 |
| Top-5% lift | **4.9×** (catches 24.6% of all real churners) |
| Top-10% lift | **3.8×** (catches 38.4% of all real churners) |
| Top-20% lift | 2.8× (catches 55.7%) |
| Bake-off ordering | tuned LR > default LR > tuned HistGBM > default HistGBM > tuned XGBoost > default XGBoost > Random Forest |
| Family choice vs tuning contribution | Family choice dominates tuning by ~5-10× at this dataset size |
| Fairness concerns | Modest calibration gap on the **casual engagement cohort** (Brier delta above auto-flag threshold). No under-service or recall-parity violations across other segments. |

**Why HistGBM over XGBoost (both trees):** HistGBM beats XGBoost on PR-AUC in the bake-off (both default and Optuna-tuned), has identical tree-model production properties (SHAP TreeExplainer, native missing-value handling, noise tolerance), and drops the external `xgboost` dependency (sklearn-only).

**Why not tuned LR (raw-metric winner):** LR is competitive on PR-AUC on this synthetic dataset (Phase 3 feature engineering captured most of the non-linearity), but lacks tree-model production properties — no SHAP richness, needs imputation for missing values, more brittle on noisier real data.

---

## Part 2: Decision rule (v1) — what we ship

### What was built

- **EV formula** (`06_decision_rule.py`, `src/decisions/policy.py`):
  ```
  EV(user, lever) = P(churn) × uplift(lever) × LTV(tier) − cost(lever)
  ```
  Target user if best available lever has positive EV; sort by EV desc, cumulative-cost cutoff at budget cap; apply 5% premium-upgrade guardrail.
- **Intervention menu** — 3 tactical levers with PM-curated cost + assumed uplift:
  - `curated_playlist` ($1 cost, 5% assumed uplift)
  - `credit_5` ($5, 15%)
  - `premium_upgrade` ($12, 25%)
- **LTV per tier** (`src/decisions/ltv.py`) — derived from Kaplan-Meier survival curves via restricted mean survival time (24-month horizon), NOT ballpark defaults:
  - Basic $200, Standard $315, Premium $435
- **Budget sweep** — 16-point curated grid from $2k to $500k, shows ROI-vs-total-EV trade-off
- **Sensitivity analysis** — ±50% uplift sweep + 12-60 month LTV horizon sweep
- **Hero figure + Streamlit app** (`07_hero_figure.py`, `app/streamlit_app.py`) — visual + interactive delivery

### Key findings — v0 vs v1

| Metric | v0 (blanket m11) | v1 (targeted, $200k ceiling) | Δ |
|---|---:|---:|---:|
| Users contacted | 1,587 | 10,933 | +9,346 (**6.9×**) |
| Total cost | $7,935 | $30,162 | +$22,227 |
| Expected retained revenue | $3,137 | $49,249 | **+$46,112 (15.7×)** |
| Net expected value | −$4,798 | **+$19,087** | **+$23,885** |
| ROI | 0.40× | 1.64× | +1.24× |

### Key findings — ROI-vs-EV trade-off (budget sweep)

| Budget | Users | Cost | Net EV | ROI | Note |
|---:|---:|---:|---:|---:|---|
| $5k | 416 | $5k | $6.2k | **2.24×** | peak ROI |
| $7.5k | 625 | $7.5k | $7.9k | 2.06× | last budget clearing 2.0× |
| $10k | 833 | $10k | $9.2k | 1.92× | |
| $30k | 10,771 | $30k | **$19.1k** | 1.64× | max total EV (policy saturates) |
| $50k–$200k | 10,933 | $30k | $19.1k | 1.63× | dormant — no more positive-EV users |

**Recommended operating budget:** $30k (max total impact). Governance ceiling: $200k (circuit-breaker). If finance requires 2.0× ROI hard target, $7.5k is the largest budget that clears it (accepting $7.9k EV instead of $19.1k).

### Key findings — sensitivity

| Uplift scale | ROI |
|---:|---:|
| 0.5× (50% weaker) | 1.43× (still beats v0) |
| 1.0× (baseline) | 1.63× |
| 1.5× (50% stronger) | 1.80× (plateau — more uplift adds users, not per-user EV) |

- **Uplift ±50%:** targeted always beats blanket.
- **LTV horizon 12-60mo:** direction of recommendation unchanged.
- **2.0× ROI at $200k is NOT reachable via uplift alone** — even at 1.5× the assumed uplift, ROI plateaus at 1.80×. Getting to 2.0× requires either a smaller budget (trade impact for ROI) or a stronger model (event-stream features).

### Key findings — lever mix

| Lever | Users targeted | % of targeted | Total spend | % of budget |
|---|---:|---:|---:|---:|
| `curated_playlist` | 7,608 | **69.6%** | $7,608 | 25.2% |
| `credit_5` | 2,478 | 22.7% | $12,390 | **41.1%** |
| `premium_upgrade` | 847 | 7.7% | $10,164 | 33.7% |

`curated_playlist` dominates by user count (cheapest lever = lowest EV break-even = broadest capture); `credit_5` dominates by dollar spend (5× cost per user). Two views tell different stories to the retention team vs the CFO.

---

## Part 3: Uplift enhancement (v2) — future work

### What was built

- **Randomized experimental data** (`src/data/simulate.py`) — single-arm design: 50% control (~25k) + 50% treated with `credit_5` (~25k). Chosen over multi-arm to maximize statistical power for the one lever we causally validate.
- **Uplift meta-learner bake-off** (`08_uplift_modeling.py`) — 4 wrappers around a shared HistGradientBoosting base estimator:
  - S-learner (treatment as feature)
  - T-learner (two independent per-arm models)
  - X-learner (residual-imputation + propensity weighting)
  - ClassTransformation (single-classifier reformulation)
- **Winner selection** — Qini AUC on the held-out test set; persisted via `src/models/production.py:load_production_uplift_model()`
- **Head-to-head** (`09_policy_comparison.py`) — v0 blanket vs v1 propensity vs v2 uplift, all restricted to credit_5 for apples-to-apples, scored against the simulator's `true_uplift` ground truth on the 25k credit_5-treated subset. Includes oracle-ceiling framing.

### Key findings — v0 vs v2 (on ground-truth credit_5-treated subset)

| Policy | Users targeted (in observed) | True retained revenue | vs v0 |
|---|---:|---:|---:|
| v0 blanket m11 (credit_5 only) | 784 | $21,180 | — |
| v2 uplift-based | 14,969 | **$463,242** | **~21.9×** |

Framed against a perfect-ranker oracle (target every user with `true_uplift × LTV > $5`):
- **Oracle ceiling:** $752,420
- **v0 captures ~3%** of the ceiling
- **v2 captures ~62%** of the ceiling — closes roughly six-sevenths of the gap v0 leaves

### Key findings — v1 vs v2 (marginal step)

On the same ground-truth subset, v2 delivers **~5.5× the true retained revenue of v1** ($463k vs $84k) by catching the **persuadable middle** — users at moderate churn risk but high responsiveness that v1's high-risk-first EV threshold filters out. Precision is similar for both (~95% persuadable-hit rate); the difference is v2's volume.

### Key findings — sleeping-dog gap (real limitation)

Both v1 and v2 target sleeping dogs (users where treatment INCREASES churn) at ~5%, matching the population base rate. v2 catches persuadables well but **can't yet sharply avoid sleeping dogs** — that's the specific weakness a production A/B holdout would tighten before shipping.

### Status

**v2 is not yet shipped.** Gated on:
1. Production randomized A/B holdout to validate the ~62%-of-ceiling number on fresh data
2. Sleeping-dog rate improvement (currently at random baseline)
3. Multi-lever extension if needed (`curated_playlist` + `premium_upgrade` would each need their own experimental arm)

---

## Key design decisions + rationale

**Why HistGradientBoosting for production (not XGBoost or LR):**
Phase 4b bake-off showed HistGBM beats XGBoost on PR-AUC. Tuned LR wins the aggregate metric but lacks tree-model production properties (SHAP richness, missing-value handling, noise tolerance). HistGBM drops the external `xgboost` dependency (sklearn-only). See `reports/prep/phase_4_qa.md` for the full argument.

**Why single-arm experiment (not multi-arm):**
Original design was 5 levers × 5k treated each = fragmented statistical power per lever. Single-arm gives 25k credit_5-treated users — 5× more training data, 5× more ground-truth evaluation. Trade-off: no ground truth for other levers; v1.1 multi-lever uplift would need a new experiment.

**Why 24-month LTV horizon (not 36 or 48):**
Longer horizons inflate LTV based on KM tail estimates from a shrinking, survivor-biased sub-population (~30% of users observed to 24 months, ~5% to 48, ~1% to 60). 24 months is the point where the sample is large enough to trust AND captures the m12 anniversary churn spike. Sensitivity table in Phase 6 Section H.5 shows the full trade-off.

**Why two-layer diagnostic vs tactical lever design:**
Phase 5's SHAP output uses a rich 10-category diagnostic vocabulary for human interpretation ("re-engagement email", "white-glove support callback"); Phase 6's decision rule uses 3 operational levers the retention platform actually deploys. Rich vocabulary for humans, simple menu for automation, connected via documented crosswalk.

**Why v1 ships before v2:**
v1 replaces the blanket loss with $19.1k monthly gain immediately. v2 sleeping-dog identification is still at random baseline — needs a production A/B holdout to sharpen. Better to deliver v1 value now than optimize the storyline.

**Why "path to 2.0× ROI" is honest, not spun:**
The kickoff metric framework set 2.0× ROI as the ship bar. v1 delivers 1.64× at $200k ceiling. Rather than manipulating uplift assumptions or budget to hit the number, the memo explicitly says 2.0× isn't reachable at v1 quality; the CFO can pick a smaller operating budget (2.24× at $5k) or accept 1.64× for max total EV. Full trade-off on the page.

---

## Limitations honestly named

- **Synthetic data.** The dataset is generated by `src/data/simulate.py` with embedded ground truth. Real-world event streams (playback completion, notifications, cross-device usage) would push PR-AUC higher than 0.20 and change some findings.
- **Training-set overlap in Phase 9 head-to-head.** Both models have some in-sample presence on the evaluation subset (~80% for v1, ~14% for v2). Comparison is "each model at its best on shared users," not strictly held-out. Real deployment needs a fresh randomized holdout.
- **Uplift model only for `credit_5`.** No ground truth for other tactical levers (`curated_playlist`, `premium_upgrade`). v1's policy for those levers still uses assumed uplift.
- **Sleeping-dog identification at random baseline.** Both v1 and v2 hit sleeping dogs at ~5% (matching population base rate). Not a v1 blocker (blanket does the same) but a v2 growth area.
- **2.0× ROI target not achieved at $200k.** ROI plateaus at 1.80× even with 50% stronger assumed uplift — need stronger model, not just better assumptions.
- **PR-AUC 0.20 is modest.** Tabular snapshots cap what any model can extract; event-stream features are the biggest headroom.

See `reports/prep/future_improvements.md` for the full ranked improvement plan.

---

## Where to find each artifact

| Artifact | Path |
|---|---|
| Shipped churn model | `models/churn_model_v1.pkl` (loaded via `src.models.production.load_production_churn_model()`) |
| Shipped uplift model (v2, not yet in production) | `models/uplift_credit_5_v1.pkl` (loaded via `src.models.production.load_production_uplift_model()`) |
| MLflow tracking | `mlruns/` — 12+ runs under `streamflix_churn` experiment; Registry alias `@production` |
| Decision memo (stakeholder-facing) | `reports/decision_memo.md` |
| Hero figure | `reports/figures/07_hero_summary.png` |
| Interactive tool | `streamlit run app/streamlit_app.py` |
| Metric framework (kickoff) | `reports/metrics_framework.md` |
| Original PM brief | `reports/scenario_brief.md` |
| Interview prep pack (12 files) | `reports/prep/` |
| Tests | `tests/` — 86 pytest, GitHub Actions CI |
| Simulator | `src/data/simulate.py` |
| Decision-rule math | `src/decisions/policy.py` |
| LTV derivation | `src/decisions/ltv.py` (KM RMST @ 24mo) |
| Feature engineering | `src/features/transforms.py` |
| Model training + tuning | `src/models/train.py` |
| SHAP + intervention crosswalk | `src/models/explain.py` |
| Uplift meta-learners | `src/models/uplift.py` |
| Production paths (single source of truth) | `src/models/production.py` |

---

## How to read the notebooks

The three-part narrative maps to notebook numbering:

```
Setup (data + features)
├── 01_data_audit.py            data quality checks
├── 02_eda.py                   exploratory + KM survival + LTV derivation
└── 03_feature_engineering.py   17 engineered features

Part 1 (churn model)
├── 04_modeling.py              train default + tuned HistGBM, calibrate, ship
├── 04b_model_comparison.py     4-family bake-off audit
├── 05_shap_levers.py           SHAP + two-layer lever crosswalk
└── 05b_fairness_audit.py       segment-parity audit

Part 2 (decision rule v1)
├── 06_decision_rule.py         EV rule + budget sweep + sensitivity
└── 07_hero_figure.py           chart generator for memo + README

Part 3 (uplift enhancement v2)
├── 08_uplift_modeling.py       4 meta-learners on credit_5
└── 09_policy_comparison.py     v0 vs v1 vs v2 head-to-head
```

Read top-to-bottom for the full story; skip to Part 2 if you just want the shipping recommendation; skip to Part 3 if you want the causal-ML enhancement.
