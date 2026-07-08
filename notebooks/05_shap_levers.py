"""
Phase 5 -- SHAP Explainability + Retention Levers
====================================================

Goal: turn the churn model from a black box into a decision-support tool.
For each flagged subscriber, we explain WHY the model flagged them AND
which retention lever the PM should pull.

Two audiences:
    - Data science: proper SHAP methodology (TreeExplainer, log-odds,
      global vs local, ranked importance)
    - Product: feature -> intervention mapping (curated with the PM),
      three worked examples showing 'flag + why + what to do'

Sections:
    A. Setup -- load features, retrain XGB for SHAP (uncalibrated)
    B. Compute SHAP values (TreeExplainer)
    C. Global feature importance (mean |SHAP|)
    D. SHAP summary plot (beeswarm) -- direction + magnitude
    E. Feature dependence for top 3 drivers
    F. Local explanations -- three worked examples
    G. Feature -> intervention mapping table
    H. Verdict + handoff to Phase 6

All figures saved under reports/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import prepare_features, train_xgboost
from src.models.explain import (
    compute_shap_values, global_importance, local_explanation,
    map_to_intervention, FEATURE_INTERVENTION_MAP,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# A. Setup
# =====================================================================
print("=" * 70)
print("A. SETUP -- load model + test features")
print("=" * 70)

raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)
X, y = prepare_features(df)

# Same three-way split as Phase 4 (same seed) so we're explaining the
# same model on the same test users.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42,
)

# Retrain uncalibrated XGBoost -- SHAP needs direct access to the tree
# structure. The calibrated model is what runs in production for
# probabilities; this one is what runs SHAP.
xgb = train_xgboost(X_train, y_train)
print(f"Retrained XGBoost on train (n={len(X_train):,}) for SHAP analysis")


# =====================================================================
# B. Compute SHAP values
# =====================================================================
print("\n" + "=" * 70)
print("B. COMPUTE SHAP VALUES (TreeExplainer)")
print("=" * 70)
# Sample 2000 test users -- SHAP over the full test set is unnecessary
# and slow. Global patterns show up cleanly at n=2000.
SAMPLE_SIZE = 2000
shap_values = compute_shap_values(xgb, X_test, sample_size=SAMPLE_SIZE)
X_shap = X_test.sample(n=SAMPLE_SIZE, random_state=42)
y_shap = y_test.loc[X_shap.index]
print(f"Computed SHAP for {SAMPLE_SIZE} test users")
print(f"Base value (expected log-odds output): {shap_values.base_values[0]:.4f}")
print(f"Base value (as probability): {1/(1+np.exp(-shap_values.base_values[0])):.4f}")
# Sanity check: sum(SHAP) + base_value == raw model output
sample_pred = xgb.predict(X_shap.iloc[[0]], output_margin=True)[0]
sample_check = shap_values.values[0].sum() + shap_values.base_values[0]
print(f"Sanity check (should match): raw margin={sample_pred:.4f}, "
      f"sum(shap)+base={sample_check:.4f}")


# =====================================================================
# C. Global feature importance
# =====================================================================
print("\n" + "=" * 70)
print("C. GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
print("=" * 70)
importance = global_importance(shap_values, list(X_shap.columns), top_n=15)
print(importance.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#F6735B" if v > 0 else "#5AD8A6"
          for v in importance["mean_signed_shap"]]
ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1],
        color=colors[::-1], alpha=0.85, edgecolor="white")
ax.set_xlabel("mean |SHAP value|  (avg contribution to log-odds)")
ax.set_title("Top 15 features by SHAP importance\n"
             "(red bars push toward churn on average, green push away)",
             fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_shap_global_importance.png",
            dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/05_shap_global_importance.png")


# =====================================================================
# D. SHAP summary plot (beeswarm)
# =====================================================================
print("\n" + "=" * 70)
print("D. SHAP SUMMARY PLOT (beeswarm)")
print("=" * 70)
# Beeswarm: one row per feature, one dot per user, x-position = SHAP,
# color = feature value (red high, blue low). Shows BOTH magnitude AND
# direction of effect across the sample.
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_shap, max_display=15, show=False,
                  plot_size=None)
plt.title("SHAP summary -- direction and magnitude of top features",
          fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_shap_beeswarm.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved -> {FIG_DIR}/05_shap_beeswarm.png")


# =====================================================================
# E. Feature dependence for top 3 drivers
# =====================================================================
print("\n" + "=" * 70)
print("E. FEATURE DEPENDENCE (top 3 drivers)")
print("=" * 70)
top_3 = importance["feature"].head(3).tolist()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, feat in zip(axes, top_3):
    idx = list(X_shap.columns).index(feat)
    x = X_shap.iloc[:, idx].values
    s = shap_values.values[:, idx]
    ax.scatter(x, s, alpha=0.35, s=12, color="#5B8FF9")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{feat}  (feature value)")
    ax.set_ylabel("SHAP contribution (log-odds)")
    ax.set_title(feat, fontweight="bold", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Feature dependence -- how SHAP changes with feature value",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_shap_dependence.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/05_shap_dependence.png")
print(f"\nTop 3 drivers:")
for feat in top_3:
    print(f"  - {feat}")


# =====================================================================
# F. Local explanations -- three worked examples
# =====================================================================
print("\n" + "=" * 70)
print("F. LOCAL EXPLANATIONS -- three worked examples")
print("=" * 70)

pred_proba = xgb.predict_proba(X_shap)[:, 1]
# Pick three example indices at different risk levels
# 1. Highest-risk user (model most confident of churn)
# 2. Marginal case near decision boundary
# 3. Confidently retained user (control example)
example_indices = [
    ("HIGH RISK   ", int(np.argmax(pred_proba))),
    ("MARGINAL    ", int(np.argmin(np.abs(pred_proba - 0.15)))),
    ("LOW RISK    ", int(np.argmin(pred_proba))),
]

for label, idx in example_indices:
    p = pred_proba[idx]
    actual = int(y_shap.iloc[idx])
    print(f"\n[{label}]  P(churn)={p:.3f}  actual={actual}")
    local = local_explanation(shap_values, X_shap, list(X_shap.columns),
                              idx=idx, top_n=6)
    print(local.to_string(index=False))

    # Attach the intervention recommendation for the top RISK+ feature
    top_risk = local[local["direction"] == "RISK+"].head(1)
    if not top_risk.empty:
        feat = top_risk["feature"].iloc[0]
        lever = map_to_intervention(feat)
        print(f"  Top intervention lever: {lever['lever']}  (cost=${lever['cost']:.0f})")
        print(f"  Reason: {lever['note']}")


# =====================================================================
# G. Feature -> intervention mapping table
# =====================================================================
print("\n" + "=" * 70)
print("G. FEATURE -> INTERVENTION MAPPING")
print("=" * 70)
# This is the bridge between model and decision rule.
lever_rows = []
for feat, spec in FEATURE_INTERVENTION_MAP.items():
    lever_rows.append({
        "feature": feat,
        "lever": spec["lever"],
        "cost": spec["cost"],
        "note": spec["note"],
    })
lever_df = pd.DataFrame(lever_rows)
print(lever_df.to_string(index=False))


# =====================================================================
# H. Verdict + Phase 6 handoff
# =====================================================================
print("\n" + "=" * 70)
print("H. PHASE 5 VERDICT")
print("=" * 70)
top_5 = importance.head(5)["feature"].tolist()
print(f"\nTop 5 SHAP drivers overall:")
for feat in top_5:
    lever = map_to_intervention(feat)
    print(f"  - {feat:<35}  lever: {lever['lever']}")
print(
    f"\nOf the top 15 features, "
    f"{sum(1 for f in importance['feature'] if map_to_intervention(f)['cost'] > 0)}"
    f" have an actionable intervention lever."
)
print("\nHandoff to Phase 6 (cost-aware decision rule):")
print("  - Per-user P(churn) from the calibrated XGBoost")
print("  - Per-user top SHAP driver from this notebook")
print("  - Intervention cost + expected uplift from the lever mapping above")
print("  - LTV by plan tier from the scenario brief")
print("\nDecision rule = argmax over levers of (P(churn) x uplift x LTV - cost).")
