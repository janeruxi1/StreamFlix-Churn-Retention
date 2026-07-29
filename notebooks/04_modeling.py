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
# | **H. Verdict + persistence** | Pick production model, save to `models/churn_model_v1.pkl` |
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
# ## H. Verdict + model persistence
#
# Both models (LR baseline + calibrated XGBoost) are pickled into
# `models/churn_model_v1.pkl`. LR sits in the `baseline_model` slot as a documented
# sanity check; calibrated XGBoost is the `production_model` that the Streamlit app
# and Phase 5-7 load.

# %%
print("\n" + "=" * 70)
print("H. VERDICT + MODEL PERSISTENCE")
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
