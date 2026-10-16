# StreamFlix — detailed reference

Moved out of the README to keep it scannable. See [README](../README.md) for the summary.

## 🗂️ Project Structure

```
streamflix-churn-retention/
├── README.md                        # You are here
├── LICENSE                          # MIT
├── data/
│   ├── README.md                    # Schema & ground truth for BOTH files below
│   ├── subscribers.csv              # v3 baseline (28 cols) — Phase 4-7
│   └── subscribers_experiment.csv   # v4 + treatment layer (33 cols) — Phase 8
├── notebooks/                       # Analyses and figure generators
│   ├── 01_data_audit.py / .ipynb        # Schema + distributions + signal screen
│   ├── 02_eda.py / .ipynb               # Segment rates + survival curves + landmark analysis
│   ├── 03_feature_engineering.py / .ipynb  # Reusable feature transforms
│   ├── 04_modeling.py / .ipynb          # LR baseline → HistGBM + calibration (v1 model)
│   ├── 04b_model_comparison.py / .ipynb # 4-family bake-off + Optuna tuning (audit)
│   ├── 05_shap_levers.py / .ipynb       # SHAP → actionable retention levers (Part 1)
│   ├── 05b_fairness_audit.py / .ipynb   # Segment performance + calibration parity (Part 1 validation)
│   ├── 06_decision_rule.py / .ipynb     # Cost-aware policy under the budget + sweep (v1 policy — Part 2)
│   ├── 07_hero_figure.py / .ipynb       # Builds reports/figures/07_hero_summary.png (Part 2)
│   ├── 08_uplift_modeling.py / .ipynb   # Experiment ATE + uplift learners (S/T/X/CT) + Qini (Part 3)
│   └── 09_policy_comparison.py / .ipynb # Blanket vs v1 vs v2 at equal spend vs ground truth (Part 3)
├── src/                             # Reusable, tested modules
│   ├── data/
│   │   ├── simulate.py              # Synthetic data generator
│   │   └── loader.py
│   ├── features/
│   │   └── transforms.py            # RFM, behavioral, billing aggregates
│   ├── models/
│   │   ├── train.py                 # Train + persist
│   │   ├── evaluate.py              # ROC/PR/Brier/calibration + Qini AUC
│   │   ├── explain.py               # SHAP utilities
│   │   ├── uplift.py                # T/S/X-learners for causal uplift
│   │   ├── tracking.py              # MLflow context + Registry promotion
│   │   └── production.py            # Single source of truth for shipped models
│   └── decisions/
│       ├── ltv.py                   # Survival curves -> LTV per tier
│       └── policy.py                # Cost-aware decision rule (shadow-price budget allocation)
├── scripts/
│   └── bootstrap_churn_ci.py        # Bootstrap intervals for test metrics
├── tests/                           # pytest unit tests
├── app/
│   └── streamlit_app.py             # Decision-support UI
├── reports/
│   ├── scenario_brief.md            # PM brief
│   ├── metrics_framework.md         # Metric tier definitions
│   ├── decision_memo.md             # Stakeholder recommendation
│   ├── metric_confidence_intervals.md  # Bootstrap intervals (scripts/)
│   └── figures/                     # Output PNGs
├── pytest.ini
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/data/simulate.py                  # generate both CSVs (baseline + experiment)
python notebooks/04_modeling.py              # train + persist the churn model (uses baseline)
python notebooks/08_uplift_modeling.py      # train + persist the uplift model (uses experiment)
pytest tests/                                # run the 93 unit tests
streamlit run app/streamlit_app.py           # launch the decision-support app
```

---

## 🎮 Interactive demo (Streamlit)

The retention team can interact with the policy without touching the notebooks:

```bash
streamlit run app/streamlit_app.py
```

Two tabs:
- **Policy overview** — KPI row (users targeted, cost, net EV, ROI), head-to-head against the current blanket-m11 baseline, lever mix
- **Per-user lookup** — pick a subscriber ID, see the risk score, diagnostic features, and recommended lever

Sidebar controls let anyone poke at the budget, per-lever cost, uplift assumptions, and blanket-baseline parameters. The app imports directly from `src/`, so the math is the same as the notebooks and is protected by the same 93 unit tests. See [`app/README.md`](../app/README.md) for deployment to Streamlit Community Cloud.

**Live demo:** [https://janeruxi1-streamflix-churn-retention.streamlit.app/](https://janeruxi1-streamflix-churn-retention.streamlit.app/)

---

## 📈 Experiment tracking (MLflow)

Every model run — across `notebooks/04_modeling.py` (production pipeline), `notebooks/04b_model_comparison.py` (bake-off + tuning), and `notebooks/08_uplift_modeling.py` (causal uplift bake-off) — is wrapped in an MLflow context (`src/models/tracking.py`) that logs parameters, metrics, and the fitted model. Twelve runs land in the local SQLite store (`mlflow.db`): from Phase 4 — `lr_baseline`, `hist_gbm_default`, `hist_gbm_tuned`, and the shipped `hist_gbm_tuned_calibrated`; from Phase 4b's bake-off — `xgboost_default`, `hist_gbm`, `random_forest`, `lr_tuned`, `xgboost_tuned`, `hist_gbm_tuned`; and from Phase 8 — `uplift_s_learner`, `uplift_t_learner`, `uplift_x_learner`, `uplift_class_transform`. Launch the tracking UI with `mlflow ui` from the project root; runs appear under the `streamflix_churn` experiment.

```bash
python notebooks/04_modeling.py        # runs land in mlruns/
mlflow ui                              # localhost:5000 — compare runs
```

**Model Registry — production promotion.** Logging models per run isn't the same as *deploying* them. Once a notebook picks a winner, `register_production_model()` (in `src/models/tracking.py`) promotes it to the MLflow Model Registry under a stable logical name and tags it with the `@production` alias:

| Registered name | Promoted by | Purpose |
|---|---|---|
| `streamflix_churn_production` | Phase 4 (calibrated HistGradientBoosting) | Per-user P(churn) for the decision rule |
| `streamflix_uplift_credit_5` | Phase 8 (Qini winner) | Experimental only; Phase 9 shows it should not ship |

Every re-run creates a new numbered version; the `@production` alias moves to point at the latest one. Downstream jobs load the current winner without hardcoding a run ID:

```python
model = mlflow.pyfunc.load_model("models:/streamflix_churn_production@production")
```

Uses the modern alias-based API (MLflow 2.9+) with graceful fallback to the legacy `Production` stage on older versions. Roll-back = re-point the alias at the previous version.

The Streamlit app still loads from the pickle in `models/churn_model_v1.pkl` (portable, self-contained, works on Streamlit Cloud without a shared tracking server). The Registry is the audit trail / deployment-time lookup for anything running inside our infra.

**Single source of truth for what ships.** `src/models/production.py` declares — in one place — the pickle path, Registry name, artifact key, and MLflow load URI for each shipped model (`CHURN_MODEL_PATH`, `CHURN_MODEL_REGISTRY_NAME`, `load_production_churn_model()`, and the corresponding `UPLIFT_MODEL_*` constants for Phase 8). The Streamlit app, Phase 4/6/7/8 notebooks, and tests all import from this module — no hardcoded pickle paths anywhere else in the codebase. This means Phase 4's `pickle.dump()`, the Streamlit app's `load_production_churn_model()`, and `register_production_model(...)` are all *guaranteed* to point at the same artifact. A consistency test suite (`tests/test_production.py`) enforces the naming conventions and error messages.

MLflow is an optional dependency; if it isn't installed, both the tracking calls AND the Registry promotion no-op silently so the pipeline still works in lightweight environments (including CI). This keeps the code path clean without forcing every reviewer to install MLflow.

---

## 🧪 Testing & CI

The `src/` modules are covered by **93 pytest unit tests** that run on every push via GitHub Actions across Python 3.10, 3.11, and 3.12. Coverage spans:

- Cost-aware decision rule (EV math, best-lever selection, shadow-price budget allocation, premium cap, blanket-baseline simulator, policy summary, sample-to-base scaling)
- LTV derivation (life-table survival from the snapshot, RMST, drift check against the shipped constants)
- Feature engineering (idempotency, no-mutation, engineered-column ranges, tenure bucketing)
- Synthetic data generator (schema, reproducibility, churn rate in band, multi-window nesting for watch hours / tickets / payment failures)
- Metrics (PR-AUC, ROC-AUC, Brier, top-K precision/recall, calibration curve)
- Model prep (drops IDs and target, one-hot categoricals, bool → int, all numeric, no nulls)
- SHAP intervention map (lever/cost lookup, unknown-feature placeholder, one-hot suffix handling)
- Uplift models (per-user prediction shape, heterogeneity capture, Qini metric sanity, uplift-based policy skips sleeping dogs)

Two of the regression tests specifically catch bugs found during development: the simulator's multi-window watch-hour scaling, and the engineered trend-ratio clip ceiling.

Run locally: `pytest tests/`

---

## 📚 Roadmap

The project is built in 13 phases, grouped into a three-part narrative:

**Setup phases** (0-3) — brief, data, EDA, features:

1. ✅ **Phase 0:** Project setup, PM brief, metric framework
2. ✅ **Phase 1:** Synthetic dataset generator + data audit
3. ✅ **Phase 2:** EDA + survival analysis (life-table survival curves, landmark analysis, sensitivity)
4. ✅ **Phase 3:** Feature engineering

**Part 1: The churn model** — predict P(churn) per user, with interpretability + fairness:

5. ✅ **Phase 4:** Modeling — LR baseline → HistGradientBoosting + calibration
6. ✅ **Phase 4b:** Model comparison — 4-family bake-off + Optuna tuning + MLflow tracking
7. ✅ **Phase 5:** SHAP — actionable retention levers
8. ✅ **Phase 5b:** Fairness / segment-parity audit on held-out rows — PR-AUC + calibration-in-the-large + targeting and recall parity across plan tier, tenure bucket, engagement cohort, country

**Part 2: Decision rule** — turn P(churn) into targeting decisions; compare **v0 vs v1**:

9. ✅ **Phase 6:** Cost-aware decision rule + budget sweep (v1 policy — churn model + assumed uplift per lever, shadow-price allocation under the budget)
10. ✅ **Phase 7:** Hero figure + decision memo + Streamlit decision-support app (v1 recommendation)

**Part 3: Experiment + uplift test** — measure credit_5 in a randomized holdout; try per-user uplift:

11. ✅ **Phase 8:** Average treatment effect with CI; S/T/X/class-transformation learners + Qini AUC
12. ✅ **Phase 9:** Blanket vs v1 vs v2 at equal spend, on held-out users, scored against the simulator's true uplift (v2 does not beat random; v1 kept)

**Delivery:**

13. ✅ **Phase 11:** Production polish (93 unit tests, GitHub Actions CI, MIT license)

---

## 💼 Key methodological choices

- **Synthetic dataset with a known treatment effect** — lets the targeting policies be scored against ground truth
- **Calibration treated as a first-class metric** — a discriminating but miscalibrated model mis-sizes the intervention budget
- **PR-AUC over ROC-AUC** as primary discrimination metric — appropriate for the imbalanced churn problem
- **Decision rule separated from model** — the model produces P(churn); the policy picks the optimal intervention per subscriber. Keeps modeling and business logic independent.
- **Budget allocation by shadow price** — when the budget binds, each subscriber gets the lever maximising `EV − λ × cost`; +33% net EV over ranking by raw EV
- **LTV from the data** — life-table survival curves per tier from the snapshot, 24-month restricted mean survival × price
- **Sensitivity analysis on budget, uplift and LTV horizon** — the recommendation holds across all of them
- **SHAP drivers mapped to candidate levers** — each flagged subscriber gets a *what might help* line; the mapping is a set of hypotheses for per-lever tests, not a causal claim
- **Uplift modelling tested, not assumed to help** — on a 25k-per-arm experiment with a 0.7pp effect, the uplift learners rank users no better than chance, so the propensity-based policy is kept and the experiment is used to measure the average effect instead

---

