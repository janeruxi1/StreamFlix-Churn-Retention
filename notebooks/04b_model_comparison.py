"""
Phase 4b -- Model Comparison (Bake-off + Hyperparameter Tuning)
================================================================

Extends Phase 4 with a broader model bake-off. Five model families +
Optuna-tuned XGBoost -- all tracked in MLflow, compared in one summary
table, winner selected on PR-AUC and Brier.

The main 04_modeling.py notebook still trains the production model.
This one shows the exploration process behind that choice.

Sections:
    A. Setup -- same splits as Phase 4
    B. Model bake-off -- LR, XGBoost, LightGBM, HistGBM, Random Forest
    C. Hyperparameter tuning -- XGBoost with Optuna (25 trials)
    D. Cross-model comparison table
    E. Production choice + verdict

Every run is logged to MLflow under experiment name 'streamflix_churn'.
Run `mlflow ui` to browse.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import (
    prepare_features, train_logistic_regression,
    train_xgboost, train_random_forest, train_hist_gbm,
    train_lightgbm, tune_xgboost_optuna, calibrate_xgboost,
)
from src.models.evaluate import compute_metrics
from src.models.tracking import mlflow_run

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# A. Setup -- same splits as Phase 4
# =====================================================================
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


# =====================================================================
# B. Model bake-off
# =====================================================================
print("\n" + "=" * 70)
print("B. MODEL BAKE-OFF (5 families)")
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


print("\nTraining 5 model families (~2-3 min total)...")
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

# LightGBM is optional -- skip cleanly if not installed
try:
    lgbm = run_and_track(
        "lightgbm", train_lightgbm,
        {"model_type": "lightgbm", "n_estimators": 300, "max_depth": 5,
         "learning_rate": 0.05},
    )
except (ImportError, OSError) as e:
    # LightGBM has known Windows C++ backend issues (access violations).
    # Skip cleanly so the rest of the bake-off still runs.
    print(f"  lightgbm                  SKIPPED ({type(e).__name__}: {e})")


# =====================================================================
# C. Hyperparameter tuning -- XGBoost with Optuna
# =====================================================================
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


# =====================================================================
# D. Cross-model comparison table
# =====================================================================
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


# =====================================================================
# E. Production choice + verdict
# =====================================================================
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
