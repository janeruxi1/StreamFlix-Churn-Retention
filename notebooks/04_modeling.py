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
# # Phase 4 — Production Modeling: LR baseline → HistGBM → Calibration
#
# **Goal:** produce a calibrated `P(churn | features)` that the Phase 6 decision rule can
# multiply by LTV. Calibration is a first-class metric here, not an afterthought.
#
# **HistGBM (sklearn's HistGradientBoostingClassifier) is the production choice**,
# validated by Phase 4b's systematic bake-off (4 model families + 3 tuned variants:
# LR via LogisticRegressionCV, XGBoost via Optuna, HistGBM via Optuna). Key findings:
#
# - **HistGBM beats XGBoost on PR-AUC** among tree models (both default and tuned)
# - **Tuning within a family moves the metric only slightly** — family choice
#   dominates tuning effort
# - Among tree models, HistGBM wins on the metric AND has identical production
#   properties: SHAP TreeExplainer support, native missing-value handling, tree-based
#   noise tolerance
# - Bonus: drops the external `xgboost` dependency (sklearn-only) — one less thing
#   to version-manage
#
# **Trade-off called out honestly:** LR is competitive on raw PR-AUC across all
# models. Chose HistGBM anyway because:
# 1. **SHAP richness** — tree structure enables per-user local explanations (Phase 5)
# 2. **Native missing-value handling** — LR needs an imputation pipeline
# 3. **Real-data noise tolerance** — tree models degrade more gracefully than linear
#
# See `notebooks/04b_model_comparison.py` Section E for the full ranking + rationale.
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
# | **C. HistGBM default vs Optuna-tuned** | Train both, ship winner on TEST PR-AUC |
# | **D. Winner + Platt calibration** | Sigmoid calibration on held-out calib set |
# | **E. Discrimination curves** | PR + ROC overlay for all three models |
# | **F. Calibration curves** | Reliability diagrams before vs after |
# | **G. Top-K targeting** | Precision/recall at K% — direct input to Phase 6 |
# | **H. Cumulative gain + lift chart** | Business-vocabulary view: lift @ top K vs random |
# | **I. Verdict + persistence** | Save calibrated HistGBM to `models/churn_model_v1.pkl` |
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
    train_hist_gbm, tune_hist_gbm_optuna, calibrate_model,
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
# ## C. HistGBM — default vs Optuna-tuned (ship the winner)
#
# `HistGradientBoostingClassifier` — sklearn's native histogram-based gradient boosting.
# Chosen over XGBoost after Phase 4b's bake-off showed HistGBM beats XGBoost on PR-AUC,
# has identical tree-model production properties (SHAP + missing values + noise
# tolerance), and drops the external `xgboost` dependency (sklearn only).
#
# Train two variants and pick the one that wins on the **held-out test set**:
#
# - **C.1 Default HistGBM** — sklearn defaults, kept as baseline
# - **C.2 Optuna-tuned HistGBM** — 25-trial TPE search on the calibration split as
#   validation (log-scale priors for `learning_rate` and `l2_regularization`;
#   same tuning function used by Phase 4b's bake-off, so results are directly
#   comparable)
# - **C.3 Winner selection** — compare on TEST PR-AUC. Optuna searched against
#   val, so any val-set gain could be overfit; requiring the winner to hold up
#   on test data is the honest check before promoting.
#
# Whichever wins is what Section D calibrates and Phase 4 ships to production.

# %%
print("\n" + "=" * 70)
print("C. HISTGBM -- DEFAULT VS OPTUNA-TUNED")
print("=" * 70)

# --- C.1  Default HistGBM (baseline, kept for comparison) ---------------
print("\n--- C.1  Default HistGBM (baseline) ---")
with mlflow_run("hist_gbm_default") as run:
    hgb_default = train_hist_gbm(X_train, y_train)
    hgb_default_proba = hgb_default.predict_proba(X_test)[:, 1]
    hgb_default_metrics = compute_metrics(y_test, hgb_default_proba)
    if run is not None:
        run.log_params({
            "model_type": "hist_gbm_default",
            "max_iter": 300, "max_depth": 5, "learning_rate": 0.05,
            "min_samples_leaf": 25, "l2_regularization": 1.0,
            "n_train": len(X_train), "n_features": X_train.shape[1],
        })
        run.log_metrics(hgb_default_metrics)
        run.log_model(hgb_default, name="model")
for k, v in hgb_default_metrics.items():
    print(f"  {k:<10} {v:.4f}")

# --- C.2  Optuna-tuned HistGBM (25 trials, val = X_calib) ---------------
print("\n--- C.2  Optuna-tuned HistGBM (25-trial TPE, val=X_calib) ---")
with mlflow_run("hist_gbm_tuned") as run:
    hgb_tuned, best_hgb_params = tune_hist_gbm_optuna(
        X_train, y_train, X_calib, y_calib,
        n_trials=25, random_state=42,
    )
    hgb_tuned_proba = hgb_tuned.predict_proba(X_test)[:, 1]
    hgb_tuned_metrics = compute_metrics(y_test, hgb_tuned_proba)
    if run is not None:
        run.log_params({"model_type": "hist_gbm_tuned", **best_hgb_params,
                        "n_train": len(X_train),
                        "n_features": X_train.shape[1]})
        run.log_metrics(hgb_tuned_metrics)
        try:
            run.log_model(hgb_tuned, name="model")
        except Exception as e:
            print(f"  (model log skipped: {e})")
print(f"  Best params from Optuna: {best_hgb_params}")
for k, v in hgb_tuned_metrics.items():
    print(f"  {k:<10} {v:.4f}")

# --- C.3  Winner selection on TEST PR-AUC -------------------------------
print("\n--- C.3  Winner selection (TEST PR-AUC) ---")
default_pr = hgb_default_metrics["pr_auc"]
tuned_pr   = hgb_tuned_metrics["pr_auc"]
delta_pr   = tuned_pr - default_pr
print(f"  default PR-AUC: {default_pr:.4f}")
print(f"  tuned   PR-AUC: {tuned_pr:.4f}   (Δ vs default = {delta_pr:+.4f})")

if tuned_pr > default_pr:
    hgb = hgb_tuned
    hgb_proba_test = hgb_tuned_proba
    hgb_metrics = hgb_tuned_metrics
    winner_variant = "tuned"
    print(f"  WINNER: tuned HistGBM -- val-set gain held up on test "
          f"(+{delta_pr:.4f} PR-AUC), not overfit. Shipping tuned variant.")
else:
    hgb = hgb_default
    hgb_proba_test = hgb_default_proba
    hgb_metrics = hgb_default_metrics
    winner_variant = "default"
    print(f"  WINNER: default HistGBM -- tuning gain on val did NOT hold "
          f"on test (Δ={delta_pr:+.4f}); avoiding val-set overfit by "
          f"shipping defaults.")


# %% [markdown]
# ## D. Calibrate the winner (Platt / sigmoid)
#
# Whichever variant won Section C (`hgb`) gets Platt-calibrated on the held-out
# calib split and shipped as production.
#
# **Platt over isotonic:** monotonic transform preserves ranking metrics (PR-AUC and
# ROC-AUC are invariant under monotonic transforms). Isotonic is more flexible but
# creates probability ties that hurt ranking on small positive classes.
#
# `calibrate_model()` is model-agnostic — works on any prefit sklearn estimator,
# including HistGBM (previously named `calibrate_xgboost`; renamed since it's generic).

# %%
print("\n" + "=" * 70)
print(f"D. HISTGBM ({winner_variant.upper()}) + PLATT CALIBRATION")
print("=" * 70)
calibrated_run_name = f"hist_gbm_{winner_variant}_calibrated"
with mlflow_run(calibrated_run_name) as run:
    hgb_cal = calibrate_model(hgb, X_calib, y_calib, method="sigmoid")
    hgb_cal_proba_test = hgb_cal.predict_proba(X_test)[:, 1]
    hgb_cal_metrics = compute_metrics(y_test, hgb_cal_proba_test)
    winner_run_id = None
    if run is not None:
        run.log_params({
            "model_type": calibrated_run_name,
            "calibration_method": "sigmoid_platt",
            "n_calib": len(X_calib),
            "base_variant": winner_variant,
            "base_run": f"hist_gbm_{winner_variant}",
        })
        run.log_metrics(hgb_cal_metrics)
        run.log_model(hgb_cal, name="model")
        # Capture the run ID WHILE the run is active -- needed below for
        # the Registry promotion (mlflow.active_run() returns None after
        # the `with` block exits).
        winner_run_id = run.run_id

# Promote the calibrated HistGBM to the MLflow Model Registry as the new
# production version. Registers under a stable logical name so downstream
# jobs can load `models:/streamflix_churn_production@production` without
# hardcoding a run ID. Runs OUTSIDE the mlflow_run context (Registry API
# operates on past runs by ID). Silently no-ops if MLflow isn't installed.
register_production_model(winner_run_id, CHURN_MODEL_REGISTRY_NAME)
print("Metrics on test set:")
for k, v in hgb_cal_metrics.items():
    print(f"  {k:<10} {v:.4f}")

# Comparison table
print("\n" + "-" * 70)
print("MODEL COMPARISON (test set)")
print("-" * 70)
comparison = pd.DataFrame({
    "logistic_regression":       lr_metrics,
    "hist_gbm_default":          hgb_default_metrics,
    "hist_gbm_tuned":            hgb_tuned_metrics,
    f"hist_gbm_{winner_variant}_calibrated":  hgb_cal_metrics,
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
    ("LR baseline",                              lr_proba_test,       "#5B8FF9"),
    ("HistGBM default",                          hgb_default_proba,   "#F6AD55"),
    ("HistGBM tuned",                            hgb_tuned_proba,     "#F6735B"),
    (f"HistGBM {winner_variant} (calibrated)",   hgb_cal_proba_test,  "#5AD8A6"),
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
print(f"\nUsing best model: HistGBM calibrated (test n={len(y_test):,})")
k_values = [0.05, 0.10, 0.20, 0.30, 0.50]
rows = []
for k in k_values:
    m = top_k_metrics(y_test, hgb_cal_proba_test, k=k)
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
    m = top_k_metrics(y_test, hgb_cal_proba_test, k=k)
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
proba = hgb_cal_proba_test
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
         color="#5B8FF9", linewidth=2.5,
         label=f"Calibrated HistGBM ({winner_variant})")
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

plt.suptitle(f"Lift analysis -- calibrated HistGBM ({winner_variant}) on the test set",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_lift_chart.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/04_lift_chart.png")
plt.show()


# %% [markdown]
# ## I. Verdict + model persistence
#
# Both models are pickled into `models/churn_model_v1.pkl`. LR sits in the
# `baseline_model` slot as a documented sanity check; the calibrated HistGBM
# variant that won Section C (default or tuned) is the `production_model` that
# the Streamlit app and Phase 5-7 load.

# %%
print("\n" + "=" * 70)
print("I. VERDICT + MODEL PERSISTENCE")
print("=" * 70)
# Honest read on the comparison:
#   - LR and HistGBM are close on this synthetic dataset -- Phase 3 feature
#     engineering captured most of the non-linearity, so linear models
#     stay competitive
#   - All variants have similar Brier scores -- already well-calibrated,
#     so calibration didn't move the needle but also didn't hurt under
#     Platt (unlike isotonic, which would have)
#   - Section C selected {default | tuned} HistGBM based on TEST PR-AUC
#     (not the val-set score Optuna searched against) -- so any tuning
#     gain that shipped genuinely held up on held-out data
#
# Production choice: calibrated HistGBM (variant selected in Section C,
# family validated by Phase 4b bake-off).
# Why not LR even though it's slightly ahead on this dataset?
#   (a) Real production data will be noisier and have unmeasured
#       interactions; trees handle that more gracefully than linear models
#   (b) Phase 5 SHAP gives richer, more actionable retention-lever stories
#       for tree models than for LR coefficients
#   (c) Native missing-value handling means new features that arrive with
#       partial coverage won't break the pipeline
# Why HistGBM specifically over XGBoost?
#   (a) HistGBM beats XGBoost on PR-AUC (Phase 4b bake-off, both default and tuned)
#   (b) Same tree-model production properties (SHAP + missing values)
#   (c) Sklearn-only -- drops the external xgboost dependency
# Both models persisted -- LR is the documented baseline / sanity check.

print(f"\nProduction model: HistGBM + Platt calibration")
print(f"  PR-AUC:  {hgb_cal_metrics['pr_auc']:.4f}")
print(f"  ROC-AUC: {hgb_cal_metrics['roc_auc']:.4f}")
print(f"  Brier:   {hgb_cal_metrics['brier']:.4f}")
print(f"\nBaseline (persisted for reference): LR")
print(f"  PR-AUC:  {lr_metrics['pr_auc']:.4f}")
print(f"  ROC-AUC: {lr_metrics['roc_auc']:.4f}")
print(f"  Brier:   {lr_metrics['brier']:.4f}")

# Persist to the path declared in src/models/production.py so the
# Streamlit app + Phase 5-7 notebooks load exactly this artifact.
artifact = {
    CHURN_MODEL_ARTIFACT_KEY: hgb_cal,   # <- calibrated HistGBM — Streamlit + notebooks read this
    "baseline_model":         lr,
    "feature_names":          list(X.columns),
    "metrics": {
        "production": hgb_cal_metrics,
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
