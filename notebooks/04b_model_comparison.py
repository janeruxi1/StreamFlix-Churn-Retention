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
# # Phase 4b — Family Bake-off: audit behind Phase 4's XGBoost choice
#
# **Chronologically, this notebook runs BEFORE Phase 4** — it's the systematic audit
# that either confirms or overturns the XGBoost family choice. Four families
# (LR, XGBoost, HistGBM, Random Forest) plus a 25-trial Optuna-tuned XGBoost,
# all tracked in MLflow, compared on PR-AUC and Brier.
#
# **Verdict this bake-off produced:** all four families cluster within ~0.01 PR-AUC of
# each other and tuning adds another ~0.002. That's the noise floor — so the Phase 4
# XGBoost choice is defensible on performance, and the tie-breakers (SHAP richness,
# native missing-value handling, calibration ease) carry the decision.
#
# This notebook does **NOT** persist a model. Phase 4 does that with the calibrated
# winner (`models/churn_model_v1.pkl`).
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Same 60/20/20 splits as Phase 4 |
# | **B. Bake-off** | LR, XGBoost, HistGBM, Random Forest (4 families) |
# | **C. Optuna tuning** | 25-trial Bayesian search over XGBoost hyperparameters |
# | **D. Comparison** | Cross-model table + bar chart |
# | **E. Verdict** | Production choice reasoning |
#
# Every run is logged to MLflow under experiment name `streamflix_churn`. Run `mlflow ui`
# from the project root to browse.

# %%
import os
import sys
import pickle
from pathlib import Path

# Run from project root whether invoked as `python notebooks/04b_...` or
# from a Jupyter cell (which doesn't define __file__).
try:
    _project_root = Path(__file__).resolve().parents[1]
except NameError:
    _here = Path.cwd()
    _project_root = _here.parent if _here.name == "notebooks" else _here
os.chdir(_project_root)
sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import (
    prepare_features, train_logistic_regression,
    train_xgboost, train_random_forest, train_hist_gbm,
    tune_xgboost_optuna, calibrate_xgboost,
)
from src.models.evaluate import compute_metrics
from src.models.tracking import mlflow_run

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## A. Setup — same splits as Phase 4

# %%
print("=" * 70)
print("A. SETUP")
print("=" * 70)
raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)
X, y = prepare_features(df)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42,
)
print(f"train n={len(X_train):,}, calib n={len(X_calib):,}, test n={len(X_test):,}")


# %% [markdown]
# ## B. Model bake-off — 4 families
#
# LR (baseline), XGBoost (default), HistGBM (sklearn native), Random Forest (bagged).
# All logged to MLflow via the same `run_and_track` helper.

# %%
print("\n" + "=" * 70)
print("B. MODEL BAKE-OFF (4 families)")
print("=" * 70)

results = {}  # {model_name: metrics_dict}
proba_holder = {}  # {model_name: y_proba_on_test}


def run_and_track(name: str, trainer_fn, params: dict):
    """Train, evaluate, log to MLflow."""
    with mlflow_run(name) as run:
        model = trainer_fn(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        if run is not None:
            run.log_params(params)
            run.log_metrics(metrics)
            try:
                run.log_model(model, name="model")
            except Exception as e:
                print(f"  (model log skipped for {name}: {e})")
    results[name] = metrics
    proba_holder[name] = proba
    print(f"  {name:<25} PR-AUC={metrics['pr_auc']:.4f}  "
          f"ROC-AUC={metrics['roc_auc']:.4f}  Brier={metrics['brier']:.4f}")
    return model


print("\nTraining 4 model families (~2-3 min total)...")
lr = run_and_track(
    "lr_baseline", train_logistic_regression,
    {"model_type": "logistic_regression", "penalty": "l2", "C": 1.0},
)
xgb = run_and_track(
    "xgboost_default", train_xgboost,
    {"model_type": "xgboost", "n_estimators": 300, "max_depth": 5,
     "learning_rate": 0.05},
)
hist_gbm = run_and_track(
    "hist_gbm", train_hist_gbm,
    {"model_type": "sklearn_hist_gbm", "max_iter": 300, "max_depth": 5,
     "learning_rate": 0.05},
)
rf = run_and_track(
    "random_forest", train_random_forest,
    {"model_type": "random_forest", "n_estimators": 300, "max_depth": 12,
     "min_samples_leaf": 10},
)


# %% [markdown]
# ## C. Hyperparameter tuning — XGBoost with Optuna (25 trials)
#
# TPE sampler with log-scaled priors for learning rate and regularization. 25 trials is
# the sweet spot: enough for TPE to warm up + exploit, few enough to complete in ~1 min.

# %%
print("\n" + "=" * 70)
print("C. HYPERPARAMETER TUNING (Optuna, 25 trials on XGBoost)")
print("=" * 70)
try:
    with mlflow_run("xgboost_tuned") as run:
        best_xgb, best_params = tune_xgboost_optuna(
            X_train, y_train, X_calib, y_calib, n_trials=25,
        )
        best_xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
        best_xgb_metrics = compute_metrics(y_test, best_xgb_proba)
        if run is not None:
            run.log_params({"model_type": "xgboost_tuned", **best_params})
            run.log_metrics(best_xgb_metrics)
            run.log_model(best_xgb, name="model")
    results["xgboost_tuned"] = best_xgb_metrics
    proba_holder["xgboost_tuned"] = best_xgb_proba
    print(f"\n  Best params from Optuna search:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  xgboost_tuned  PR-AUC={best_xgb_metrics['pr_auc']:.4f}  "
          f"ROC-AUC={best_xgb_metrics['roc_auc']:.4f}  "
          f"Brier={best_xgb_metrics['brier']:.4f}")
except ImportError as e:
    print(f"  Optuna not installed: {e}")
    best_xgb = xgb  # fall back to default XGBoost for calibration step


# %% [markdown]
# ## D. Cross-model comparison table + bar chart

# %%
print("\n" + "=" * 70)
print("D. CROSS-MODEL COMPARISON")
print("=" * 70)
comparison = pd.DataFrame(results).T.round(4)
comparison = comparison.sort_values("pr_auc", ascending=False)
print(comparison)

# Save comparison as a figure
fig, ax = plt.subplots(figsize=(9, 5))
model_names = comparison.index.tolist()
pr_aucs = comparison["pr_auc"].values
briers = comparison["brier"].values

colors = ["#5AD8A6" if i == 0 else "#5B8FF9" for i in range(len(model_names))]
bars = ax.barh(model_names[::-1], pr_aucs[::-1], color=colors[::-1],
               edgecolor="white")
ax.set_xlabel("PR-AUC (higher is better)")
ax.set_title("Model bake-off -- PR-AUC on test set",
             fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.axvline(y_test.mean(), color="gray", linestyle=":", linewidth=1,
           label=f"random baseline (positive rate = {y_test.mean():.3f})")
ax.legend(loc="lower right")
for bar, v in zip(bars, pr_aucs[::-1]):
    ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "04b_model_comparison.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/04b_model_comparison.png")


# %% [markdown]
# ## E. Production choice + verdict
#
# Tie-break rule when PR-AUC differences are within ~0.01: lower Brier wins (calibration
# matters for Phase 6). Prefer trees over LR for SHAP richness + native missingness.

# %%
print("\n" + "=" * 70)
print("E. PRODUCTION CHOICE")
print("=" * 70)
winner_name = comparison.index[0]
winner_metrics = comparison.iloc[0].to_dict()
print(f"\nHead-to-head winner on PR-AUC: {winner_name}")
print(f"  PR-AUC: {winner_metrics['pr_auc']:.4f}")
print(f"  Brier:  {winner_metrics['brier']:.4f}")

print(f"""
Production choice reasoning:
- If the top two are within noise (~0.01 PR-AUC), tie-break on Brier
  (calibration matters for the decision rule)
- Prefer tree models over LR for production noise-tolerance and
  richer SHAP explanations
- The Phase 4 production choice (Platt-calibrated XGBoost) already
  satisfies these criteria. If the tuned XGBoost beats the default,
  swap in the tuned version + re-run calibration.

The full run log is in mlruns/ -- compare in `mlflow ui` at localhost:5000.
""")
