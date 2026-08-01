# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Phase 4 — Production Modeling: LR baseline → XGBoost → Calibration
#
# **Goal:** produce a calibrated `P(churn | features)` that the Phase 6 decision rule can
# multiply by LTV. Calibration is a first-class metric here, not an afterthought.
#
# The **XGBoost family choice is validated by Phase 4b's systematic bake-off**
# (4 families + 25-trial Bayesian tuning). This notebook takes that winner and builds the
# production-quality artifact: calibrated, evaluated on held-out data, persisted to
# `models/churn_model_v1.pkl` for the Streamlit app and Phase 5-7 to consume.
#
# **Chronologically, Phase 4b's bake-off runs BEFORE this notebook** (family selection).
# Phase 4 is numbered/presented first because it's the deployment story a PM or
# engineering partner would open. Read them in either order.
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Load engineered data, prepare features, three-way split |
# | **B. LR baseline** | Regularized logistic regression on the same features |
# | **C. XGBoost (uncalibrated)** | Boosted trees, default hyperparams |
# | **D. XGBoost + Platt calibration** | Sigmoid calibration on held-out calib set |
# | **E. Discrimination curves** | PR + ROC overlay for all three models |
# | **F. Calibration curves** | Reliability diagrams before vs after |
# | **G. Top-K targeting** | Precision/recall at K% — direct input to Phase 6 |
# | **H. Cumulative gain + lift chart** | Business-vocabulary view: lift @ top K vs random |
# | **I. Verdict + persistence** | Pick production model, save to `models/churn_model_v1.pkl` |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
from pathlib import Path

# Run from project root whether invoked as `python notebooks/04_...` or
# from a Jupyter cell (which doesn't define __file__).
try:
    _project_root = Path(__file__).resolve().parents[1]
except NameError:
    _here = Path.cwd()
    _project_root = _here.parent if _here.name == "notebooks" else _here
os.chdir(_project_root)
sys.path.insert(0, str(_project_root))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score,
)

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import (
    prepare_features, train_logistic_regression,
    train_xgboost, calibrate_xgboost,
)
from src.models.evaluate import (
    compute_metrics, top_k_metrics, calibration_curve_points,
)
from src.models.tracking import mlflow_run, register_production_model
from src.models.production import (
    CHURN_MODEL_PATH, CHURN_MODEL_ARTIFACT_KEY, CHURN_MODEL_REGISTRY_NAME,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## A. Setup — load, prepare features, three-way split

# %%
print("=" * 70)
print("A. SETUP")
print("=" * 70)

raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)
X, y = prepare_features(df)
print(f"Feature matrix: X={X.shape}, y_positive_rate={y.mean():.4f}")
print(f"Total features (post one-hot): {X.shape[1]}")

# Three-way split: train (60%) / calib (20%) / test (20%)
# Stratify on y to preserve the ~5% positive rate in every split.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42,
)  # 0.25 of 0.80 = 0.20 overall
print(f"\nSplits:")
print(f"  train: n={len(X_train):,}  positive_rate={y_train.mean():.4f}")
print(f"  calib: n={len(X_calib):,}  positive_rate={y_calib.mean():.4f}")
print(f"  test:  n={len(X_test):,}  positive_rate={y_test.mean():.4f}")


# %% [markdown]
# ## B. Logistic regression baseline
#
# Every model variant is wrapped in an MLflow run so we can compare runs in the tracking
# UI (`mlflow ui` → localhost:5000). If MLflow isn't installed, `mlflow_run()` no-ops
# and we fall through to the untracked path — keeps CI lightweight.

# %%
print("\n" + "=" * 70)
print("B. LOGISTIC REGRESSION BASELINE")
print("=" * 70)
with mlflow_run("lr_baseline") as run:
    lr = train_logistic_regression(X_train, y_train)
    lr_proba_test = lr.predict_proba(X_test)[:, 1]
    lr_metrics = compute_metrics(y_test, lr_proba_test)
    if run is not None:
        run.log_params({
            "model_type": "logistic_regression",
            "penalty": "l2", "C": 1.0, "max_iter": 2000,
            "n_train": len(X_train), "n_features": X_train.shape[1],
            "random_state": 42,
        })
        run.log_metrics(lr_metrics)
        run.log_model(lr, name="model")
print("Metrics on test set:")
for k, v in lr_metrics.items():
    print(f"  {k:<10} {v:.4f}")


# %% [markdown]
# ## C. XGBoost (uncalibrated)

# %%
print("\n" + "=" * 70)
print("C. XGBOOST (UNCALIBRATED)")
print("=" * 70)
with mlflow_run("xgboost_uncalibrated") as run:
    xgb = train_xgboost(X_train, y_train)
    xgb_proba_test = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba_test)
    if run is not None:
        run.log_params({
            "model_type": "xgboost",
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 5, "reg_lambda": 1.0,
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "n_train": len(X_train), "n_features": X_train.shape[1],
        })
        run.log_metrics(xgb_metrics)
        run.log_model(xgb, name="model")
print("Metrics on test set:")
for k, v in xgb_metrics.items():
    print(f"  {k:<10} {v:.4f}")


# %% [markdown]
# ## D. XGBoost + Platt (sigmoid) calibration
#
# **Platt over isotonic:** monotonic transform preserves ranking metrics (PR-AUC and
# ROC-AUC are invariant under monotonic transforms). Isotonic is more flexible but
# creates probability ties that hurt ranking on small positive classes.

# %%
print("\n" + "=" * 70)
print("D. XGBOOST + PLATT (SIGMOID) CALIBRATION")
print("=" * 70)
with mlflow_run("xgboost_calibrated") as run:
    xgb_cal = calibrate_xgboost(xgb, X_calib, y_calib, method="sigmoid")
    xgb_cal_proba_test = xgb_cal.predict_proba(X_test)[:, 1]
    xgb_cal_metrics = compute_metrics(y_test, xgb_cal_proba_test)
    winner_run_id = None
    if run is not None:
        run.log_params({
            "model_type": "xgboost_calibrated",
            "calibration_method": "sigmoid_platt",
            "n_calib": len(X_calib),
            "base_run": "xgboost_uncalibrated",
        })
        run.log_metrics(xgb_cal_metrics)
        run.log_model(xgb_cal, name="model")
        # Capture the run ID WHILE the run is active -- needed below for
        # the Registry promotion (mlflow.active_run() returns None after
        # the `with` block exits).
        winner_run_id = run.run_id

# Promote the calibrated XGBoost to the MLflow Model Registry as the new
# production version. Registers under a stable logical name so downstream
# jobs can load `models:/streamflix_churn_production@production` without
# hardcoding a run ID. Runs OUTSIDE the mlflow_run context (Registry API
# operates on past runs by ID). Silently no-ops if MLflow isn't installed.
register_production_model(winner_run_id, CHURN_MODEL_REGISTRY_NAME)
print("Metrics on test set:")
for k, v in xgb_cal_metrics.items():
    print(f"  {k:<10} {v:.4f}")

# Comparison table
print("\n" + "-" * 70)
print("MODEL COMPARISON (test set)")
print("-" * 70)
comparison = pd.DataFrame({
    "logistic_regression":  lr_metrics,
    "xgboost_uncal":        xgb_metrics,
    "xgboost_calibrated":   xgb_cal_metrics,
}).round(4)
print(comparison)


# %% [markdown]
# ## E. Discrimination curves — PR + ROC overlay
#
# PR-AUC is the primary metric (5% positive class); ROC-AUC is the lingua franca for
# comparison against published benchmarks.

# %%
print("\n" + "=" * 70)
print("E. DISCRIMINATION CURVES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = [
    ("LR baseline",          lr_proba_test,     "#5B8FF9"),
    ("XGBoost (uncalibrated)", xgb_proba_test,   "#F6735B"),
    ("XGBoost (calibrated)",  xgb_cal_proba_test, "#5AD8A6"),
]

# PR curve
ax = axes[0]
for name, proba, color in models:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f"{name} (AP={ap:.3f})")
ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1,
           label=f"baseline (positive rate={y_test.mean():.3f})")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision-Recall curves\n(primary metric -- imbalanced classes)",
             fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# ROC curve
ax = axes[1]
for name, proba, color in models:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba) if False else compute_metrics(y_test, proba)["roc_auc"]
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
        label="random")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("ROC curves\n(secondary metric)", fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.4)

# Fix the import use above
from sklearn.metrics import roc_auc_score
plt.suptitle("Discrimination: how well does the model rank churners above non-churners?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_discrimination_curves.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/04_discrimination_curves.png")


# %% [markdown]
# ## F. Calibration curves — reliability diagrams
#
# Does `P(churn) = 0.20` really mean 20% of those users churn? The reliability diagram
# uses quantile bins (equal-population, not equal-width — critical for imbalanced classes
# where equal-width bins are mostly empty).

# %%
print("\n" + "=" * 70)
print("F. CALIBRATION CURVES (reliability diagrams)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 7))
for name, proba, color in models:
    cal_pts = calibration_curve_points(y_test, proba, n_bins=10)
    ax.plot(cal_pts["mean_pred"], cal_pts["frac_positive"],
            marker="o", linewidth=2, color=color, label=name)
ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
        linewidth=1, label="perfectly calibrated")
ax.set_xlabel("mean predicted probability (per quantile bin)")
ax.set_ylabel("actual fraction positive in bin")
ax.set_title("Calibration -- does P(churn)=0.20 mean 20% really churn?",
             fontweight="bold")
ax.set_xlim(0, max(0.6, ax.get_xlim()[1]))
ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_calibration_curve.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/04_calibration_curve.png")

# Numeric calibration summary -- Brier and log loss (lower = better)
print("\nCalibration quality (lower is better):")
print(f"  {'Model':<25} {'Brier':>10} {'Log loss':>10}")
for name, proba, _ in models:
    bs = compute_metrics(y_test, proba)["brier"]
    ll = compute_metrics(y_test, proba)["log_loss"]
    print(f"  {name:<25} {bs:>10.4f} {ll:>10.4f}")


# %% [markdown]
# ## G. Top-K targeting analysis
#
# Maps to the decision rule: *"if Retention can only contact K% of users, what fraction
# reached are real churners (precision), and what fraction of actual churners do we
# catch (recall)?"* Direct input to Phase 6.

# %%
print("\n" + "=" * 70)
print("G. TOP-K TARGETING ANALYSIS")
print("=" * 70)
print(f"\nUsing best model: XGBoost calibrated (test n={len(y_test):,})")
k_values = [0.05, 0.10, 0.20, 0.30, 0.50]
rows = []
for k in k_values:
    m = top_k_metrics(y_test, xgb_cal_proba_test, k=k)
    rows.append({
        "top_k_pct":      f"{int(k*100):>3d}%",
        "n_targeted":     m["k_count"],
        "precision_at_k": round(m["precision_at_k"], 4),
        "recall_at_k":    round(m["recall_at_k"], 4),
    })
top_k_df = pd.DataFrame(rows)
print(top_k_df.to_string(index=False))

# Visualize: precision/recall vs K curve
ks = np.linspace(0.01, 1.0, 50)
precisions, recalls = [], []
for k in ks:
    m = top_k_metrics(y_test, xgb_cal_proba_test, k=k)
    precisions.append(m["precision_at_k"])
    recalls.append(m["recall_at_k"])

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(ks * 100, precisions, color="#5B8FF9", linewidth=2.5,
         label="precision @ K (% of contacts who really churn)")
ax1.plot(ks * 100, recalls, color="#F6735B", linewidth=2.5,
         label="recall @ K (% of churners reached)")
ax1.axvline(10, color="gray", linestyle=":", linewidth=1.5,
            label="top 10% reference")
ax1.set_xlabel("K (% of users targeted, sorted by P(churn) descending)")
ax1.set_ylabel("precision / recall")
ax1.set_title("Top-K targeting tradeoff -- input to Phase 6 decision rule",
              fontweight="bold")
ax1.set_xlim(0, 100); ax1.set_ylim(0, 1)
ax1.legend(loc="center right")
ax1.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_top_k_targeting.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/04_top_k_targeting.png")


# %% [markdown]
# ## H. Cumulative gain + lift chart
#
# The classical marketing-analytics view of Section G's top-K story, translated from
# DS vocabulary (precision/recall) into business vocabulary (gain/lift). What a PM
# will point at during a review.
#
# - **Cumulative gain curve** — "if we target the top K% by score, what fraction of
#   true churners do we catch?" A perfect model captures all churners after targeting
#   only `n_positives / n_total` of the population (~5.4% in our case).
# - **Lift chart** — "how many times better than random is targeting the top K%?"
#   Lift @10% = 3.5 means the top decile has 3.5× the churn concentration of the
#   overall population.

# %%
print("\n" + "=" * 70)
print("H. CUMULATIVE GAIN + LIFT CHART")
print("=" * 70)

# Sort users by predicted probability descending
proba = xgb_cal_proba_test
order = np.argsort(-proba)
y_sorted = y_test.values[order]

n_total = len(y_sorted)
n_pos = int(y_sorted.sum())
base_rate = n_pos / n_total

# Cumulative curves
cumulative_positives = np.cumsum(y_sorted)
cumulative_gain = cumulative_positives / n_pos               # fraction caught
cumulative_share = np.arange(1, n_total + 1) / n_total       # fraction targeted

# Per-decile lift table
decile_rows = []
for d in range(1, 11):
    k_frac = d / 10
    k_count = int(np.ceil(n_total * k_frac))
    n_caught = int(y_sorted[:k_count].sum())
    gain = n_caught / n_pos if n_pos else 0
    lift = gain / k_frac if k_frac > 0 else 0
    decile_rows.append({
        "decile":     f"top {int(k_frac*100)}%",
        "k_count":    k_count,
        "n_caught":   n_caught,
        "gain":       round(gain, 3),
        "lift":       round(lift, 2),
    })
lift_df = pd.DataFrame(decile_rows)
print("\nLift by cumulative decile:")
print(lift_df.to_string(index=False))

# Business-language summary
print(f"\nKey targeting summary (test set, n={n_total:,}, {n_pos:,} true churners):")
for k_pct in [0.05, 0.10, 0.20]:
    k_count = int(np.ceil(n_total * k_pct))
    n_caught = int(y_sorted[:k_count].sum())
    gain = n_caught / n_pos
    lift = gain / k_pct
    print(f"  Target top {int(k_pct*100):>3d}% ({k_count:>5,} users)  ->  "
          f"catch {n_caught:>4,} churners ({gain:.1%} of all)  ->  lift = {lift:.2f}x")

# Two-panel figure: gain curve + lift bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# LEFT: cumulative gain curve
ax1.plot(cumulative_share * 100, cumulative_gain * 100,
         color="#5B8FF9", linewidth=2.5, label="Calibrated XGBoost")
ax1.plot([0, 100], [0, 100], color="gray", linestyle="--",
         linewidth=1, label="random targeting")
# Perfect model reaches 100% gain at share = base rate
perfect_x = [0, base_rate * 100, 100]
perfect_y = [0, 100, 100]
ax1.plot(perfect_x, perfect_y, color="#5AD8A6", linestyle=":",
         linewidth=1.5, label=f"perfect model (ceiling)")
ax1.axvline(10, color="#F6AD55", linestyle=":", linewidth=1.5,
            alpha=0.7, label="top 10% ref")
ax1.set_xlabel("Share of users targeted (%)")
ax1.set_ylabel("Share of true churners caught (%)")
ax1.set_title("Cumulative gain curve\n"
              "how many churners we catch as we widen the target set",
              fontweight="bold", fontsize=11)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="lower right", fontsize=9)

# RIGHT: lift bars per decile
bar_colors = ["#5AD8A6" if l >= 2.0 else ("#5B8FF9" if l >= 1.0 else "#F6735B")
              for l in lift_df["lift"]]
bars = ax2.bar(lift_df["decile"], lift_df["lift"], color=bar_colors,
               edgecolor="white")
ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
            label="random baseline (1.0×)")
ax2.set_ylabel("Lift multiplier (vs random targeting)")
ax2.set_title("Lift by cumulative decile\n"
              "green = ≥ 2× random, blue = above random, red = below",
              fontweight="bold", fontsize=11)
for bar, v in zip(bars, lift_df["lift"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
             f"{v:.2f}×", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(range(len(lift_df)))
ax2.set_xticklabels(lift_df["decile"], rotation=30, ha="right")
ax2.set_ylim(0, max(lift_df["lift"]) * 1.15)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(loc="upper right")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Lift analysis -- calibrated XGBoost on the test set",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_lift_chart.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/04_lift_chart.png")
plt.show()


# %% [markdown]
# ## I. Verdict + model persistence
#
# Both models (LR baseline + calibrated XGBoost) are pickled into
# `models/churn_model_v1.pkl`. LR sits in the `baseline_model` slot as a documented
# sanity check; calibrated XGBoost is the `production_model` that the Streamlit app
# and Phase 5-7 load.

# %%
print("\n" + "=" * 70)
print("I. VERDICT + MODEL PERSISTENCE")
print("=" * 70)
# Honest read on the comparison:
#   - LR and XGBoost are within 1-2 PR-AUC points of each other -- noise
#     level given a 5% positive class and a 10k test set
#   - All three models have similar Brier scores (~0.047) -- already
#     well-calibrated, so calibration didn't move the needle but also
#     didn't hurt under Platt (unlike isotonic, which would have)
#   - LR baseline is genuinely competitive -- a sign that Phase 3 feature
#     engineering captured most of the non-linearity manually
#
# Production choice: calibrated XGBoost.
# Why not LR even though it's slightly ahead on this dataset?
#   (a) Real production data will be noisier and have unmeasured
#       interactions; trees handle that more gracefully than linear models
#   (b) Phase 5 SHAP gives richer, more actionable retention-lever stories
#       for tree models than for LR coefficients
#   (c) Native missing-value handling means new features that arrive with
#       partial coverage won't break the pipeline
# Both models persisted -- LR is the documented baseline / sanity check.

print(f"\nProduction model: XGBoost + Platt calibration")
print(f"  PR-AUC:  {xgb_cal_metrics['pr_auc']:.4f}")
print(f"  ROC-AUC: {xgb_cal_metrics['roc_auc']:.4f}")
print(f"  Brier:   {xgb_cal_metrics['brier']:.4f}")
print(f"\nBaseline (persisted for reference): LR")
print(f"  PR-AUC:  {lr_metrics['pr_auc']:.4f}")
print(f"  ROC-AUC: {lr_metrics['roc_auc']:.4f}")
print(f"  Brier:   {lr_metrics['brier']:.4f}")

# Persist to the path declared in src/models/production.py so the
# Streamlit app + Phase 5-7 notebooks load exactly this artifact.
artifact = {
    CHURN_MODEL_ARTIFACT_KEY: xgb_cal,   # <- key that Streamlit + notebooks read
    "baseline_model":         lr,
    "feature_names":          list(X.columns),
    "metrics": {
        "production": xgb_cal_metrics,
        "baseline":   lr_metrics,
    },
    "training_meta": {
        "n_train": len(X_train),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "positive_rate": float(y.mean()),
    },
}
CHURN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CHURN_MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)
print(f"\nSaved -> {CHURN_MODEL_PATH}")
print(f"Registered as -> {CHURN_MODEL_REGISTRY_NAME}@production (via MLflow Registry)")
print(f"\nReady for Phase 5 (SHAP -- actionable retention levers).")
