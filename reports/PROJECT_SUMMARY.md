# Project Summary — StreamFlix Churn Retention

**What was built and what was found.** A one-page catalog of the project, with every headline number traced to the notebook that produces it. Money figures are per month for the full ~2.1M-subscriber base, estimated on a 50,000-subscriber sample (×42). The data is synthetic.

---

## 60-second overview

StreamFlix sends a $5 credit to every subscriber reaching month 11 (~67k a month). This project replaces that with a cost-aware targeting policy, in three parts:

- **Part 1 — Churn model:** predict 30-day churn per subscriber (calibrated gradient boosting).
- **Part 2 — Decision rule (v1, ships):** `P(churn) × uplift × LTV − cost` per lever, with the budget allocated by shadow price.
- **Part 3 — Experiment and uplift model (v2, not shipped):** a randomized holdout of the $5 credit measures its average effect; an uplift model trained on it fails to beat random targeting.

**Headline.** v1 turns a **−$252k** blanket campaign into **+$174k** net expected value at the $200k budget (ROI 1.87×), a **+$426k** monthly swing on $133k less spend.

---

## Headline numbers

| | Blanket m11 (today) | v1 targeted | Source |
|---|---:|---:|---|
| Subscribers contacted | 66,654 | 118,986 | 06, Sections E–F |
| Spend | $333k | $200k | 06 |
| Expected retained revenue | $82k | $374k | 06 |
| **Net expected value** | **−$252k** | **+$174k** | 06 |
| ROI | 0.25× | 1.87× | 06 |

| Part 3 head-to-head (credit only, $200k, 3,010 held-out users, scored against true effects) | True net value | Source |
|---|---:|---|
| v1: churn risk × 15% | +$130k | 09, Section C |
| v2: uplift model | −$132k | 09 |
| Random | −$149k | 09 |
| Oracle | +$233k | 09 |

---

## Part 1 — Churn model

**Built:**
- **Notebook 03:** 17 engineered features (engagement trend, recency, lifecycle, composites).
- **Notebook 04b:** a four-family bake-off (logistic regression, XGBoost, HistGBM, random forest), each Optuna- or CV-tuned.
- **Notebook 04:** tuned HistGBM with Platt calibration, persisted via `src/models/production.py`. The tuned variant was chosen before scoring the test set.
- **Notebook 05:** SHAP with a driver → candidate-lever map (`src/models/explain.py`).
- **Notebook 05b:** fairness and calibration audit on held-out rows.

| Finding | Value |
|---|---|
| PR-AUC (test, 534 churners in 10k) | **0.177**; bootstrap 95% interval about ±0.03; random = 0.053 |
| ROC-AUC / Brier | 0.750 / 0.047 (base-rate Brier 0.051) |
| Lift: top 5% / 10% / 20% | 4.6× / 3.5× / 2.7× (catch 23% / 35% / 53% of churners) |
| Model families | LR 0.178, tuned XGBoost 0.178, tuned HistGBM 0.177: statistically tied. HistGBM kept for practical reasons, LR is the fallback |
| Calibration by segment | Within ±13% everywhere except **month 12**: predicted 4.9% vs observed 8.3% (p = 0.014) |
| Targeting by segment | Low for low-churn groups (heavy viewers, tenure 25+); follows churn and LTV, not miscalibration |

---

## Part 2 — Decision rule (v1)

**Built:**
- **`src/decisions/ltv.py`:** LTV per tier from life-table survival curves (monthly hazard = churn rate at each tenure), 24-month restricted mean survival × price: **Basic $109, Standard $187, Premium $296**.
- **`src/decisions/policy.py`:** EV per subscriber × lever, then `build_policy`, which allocates a binding budget by shadow price λ (each subscriber gets the lever maximising `EV − λ × cost`), with the Premium-upgrade capacity cap inside the same loop.
- **Scale:** the brief's $200k budget and 10k Premium cap are scaled to the sample, and results are scaled back.

| Finding | Value |
|---|---|
| Positive-EV subscribers | 9.6% of the base; covering them all would cost $415k |
| Shadow price at $200k | λ = 0.31 (the last dollar returns $1.31) |
| Allocation method | Shadow price beats ranking by raw EV by **33%** ($174k vs $131k) |
| Lever mix (contacts / spend) | playlist 85% / 50%, credit 14% / 43%, Premium upgrade 1% / 7% |
| Budget for ROI ≥ 2.0× | ≤ $150k (net EV $155k) |
| Uplift ×0.5 / ×1.5 | net EV $18k / $362k; ROI 1.32× / 2.81× |
| LTV horizon 12 / 36 months | net EV $44k / $273k |
| Blanket m11 coverage | v1 contacts only 4% of m11 subscribers |

---

## Part 3 — Experiment and uplift model (v2)

**Built:**
- **Notebook 08:** experiment ATE with CI, and four uplift learners (S, T, X, class transformation) with Qini curves.
- **Notebook 09:** blanket vs v1 vs v2 vs oracle vs random at equal spend, on users neither model trained on, scored against true uplift.

| Finding | Value |
|---|---|
| Churn, control vs credit | 5.28% vs 4.61%: **−0.67pp** (95% CI 0.29–1.05pp) |
| Relative effect | **12.6%** of would-be churners retained (CI 5–20%) vs 15% assumed |
| Uplift learners | Qini ≈ 0 for all four; rank correlation with true uplift +0.02 |
| Rank correlation with true value | v1 0.82, v2 0.00 |
| Verdict | Keep v1; use the experiment to calibrate its credit uplift; don't ship v2 |

---

## Key design decisions

- **Model choice on practical grounds, not noise.** The top models are tied within the bootstrap interval, so the test set isn't used to pick between them.
- **Survival from a snapshot.** Each subscriber is seen once, so the month-t hazard is the churn rate of subscribers at tenure t. A Kaplan-Meier at-risk set of "tenure ≥ t" would treat survival through month t as observed when it wasn't, and overstate retention (S(24) of 0.92–0.96 instead of 0.26–0.44).
- **Shadow-price allocation.** Under a binding budget, the right comparison is value per dollar at the margin, not raw EV.
- **Equal-spend comparison of policies.** Comparing policies that contact different numbers of people mostly measures how many people they contact.
- **Uplift modelling tested, not assumed.** The experiment is powered for the average effect, not for per-user effects; the head-to-head shows it.

---

## Limitations

- **Synthetic data.** The ground truth is known by construction; real data would be noisier.
- **Untested uplift assumptions.** The playlist and Premium-upgrade uplifts are untested planning figures. Playlists are 85% of contacts.
- **No out-of-time validation.** Validation is a random split (60/20/20), and LTV assumes stable monthly hazards.
- **In-sample scores.** Policy scores for the whole base include the model's training rows.
- **Guardrail breached.** About 72% of spend goes to subscribers who would not have churned, against a < 30% guardrail that no useful policy can meet at a 5% base rate (see the memo).

---

## Where to find each artifact

| Artifact | Path |
|---|---|
| Decision memo | `reports/decision_memo.md` |
| PM brief, metric framework | `reports/scenario_brief.md`, `reports/metrics_framework.md` |
| Metric confidence intervals | `reports/metric_confidence_intervals.md` (`scripts/bootstrap_churn_ci.py`) |
| Hero figure | `reports/figures/07_hero_summary.png` |
| Interactive tool | `streamlit run app/streamlit_app.py` |
| Notebooks 01–09 | `notebooks/` (`.py` source, `.ipynb` generated and executed) |
| Data and ground truth | `data/README.md` |
| Structure, MLflow, tests | `docs/DETAILS.md` |

## How to read the notebooks

```
01 data audit → 02 EDA + survival → 03 features
→ 04 model → 04b bake-off → 05 SHAP → 05b fairness/calibration     (Part 1)
→ 06 decision rule → 07 hero figure                                (Part 2)
→ 08 experiment + uplift → 09 policy head-to-head                  (Part 3)
```
