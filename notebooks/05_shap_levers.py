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
# # Phase 5 — SHAP Explainability + Retention Levers
#
# **Goal:** turn the churn model from a black box into a decision-support tool. For each
# flagged subscriber, we explain **why** the model flagged them AND **which retention
# lever** the PM should pull.
#
# **Two audiences:**
# - **Data science** — proper SHAP methodology (TreeExplainer, log-odds space, global vs
#   local, ranked importance)
# - **Product** — feature → intervention mapping curated with the PM, three worked
#   examples showing *"flag + why + what to do"*
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Load features, retrain uncalibrated XGB (SHAP needs the raw booster) |
# | **B. Compute SHAP** | TreeExplainer on a 2K test sample |
# | **C. Global importance** | Mean-abs-SHAP ranked feature table + bar chart |
# | **D. Beeswarm** | Direction + magnitude across the sample |
# | **E. Dependence** | How SHAP shifts with feature value for the top 3 drivers |
# | **F. Local explanations** | Three worked examples: HIGH / MARGINAL / LOW risk |
# | **G. Diagnostic intervention map** | Feature → diagnostic lever + cost + note |
# | **H. Diagnostic → tactical crosswalk** | How the 10 diagnostic categories collapse into Phase 6's 3 operational levers |
# | **I. Verdict** | Handoff to Phase 6 decision rule |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
import warnings
from pathlib import Path

# Silence a sklearn 1.8+ FutureWarning triggered inside SHAP / other deps
# (they call the deprecated `sklearn.utils.extmath.stable_cumsum`). Will
# clear itself once those libraries release a compatible version. Not our
# code, not actionable here.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Function stable_cumsum is deprecated",
)

# Run from project root whether invoked as `python notebooks/05_...` or
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
import shap
from sklearn.model_selection import train_test_split

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import prepare_features, train_hist_gbm, tune_hist_gbm_optuna
from src.models.evaluate import compute_metrics
from src.models.explain import (
    compute_shap_values, global_importance, local_explanation,
    map_to_intervention, FEATURE_INTERVENTION_MAP,
)
from src.decisions.policy import INTERVENTION_MENU, LTV_BY_TIER

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## A. Setup — retrain uncalibrated HistGBM for SHAP
#
# The calibrated model runs in production for probabilities; this uncalibrated version
# is what runs SHAP because TreeExplainer needs direct access to the tree
# structure (the calibration wrapper hides it). Same three-way split as Phase 4 (same
# seed) so we're explaining the same model on the same test users.
#
# **Note:** After the v1.1 flip to HistGBM production, this notebook retrains
# HistGradientBoosting (not XGBoost) so SHAP explains the SAME model that ships.
# SHAP's TreeExplainer supports HistGBM since SHAP 0.35+.

# %%
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

# Retrain uncalibrated HistGBM -- SHAP needs direct access to the tree
# structure. The calibrated model is what runs in production for
# probabilities; this one is what runs SHAP.
#
# Match Phase 4's variant selection: train both default and Optuna-tuned,
# pick whichever wins on TEST PR-AUC. This way SHAP explains the SAME
# variant that ships, not just the default.
hgb_default = train_hist_gbm(X_train, y_train)
hgb_tuned, _best_params = tune_hist_gbm_optuna(
    X_train, y_train, X_calib, y_calib, n_trials=25, random_state=42,
)
default_pr = compute_metrics(
    y_test, hgb_default.predict_proba(X_test)[:, 1])["pr_auc"]
tuned_pr = compute_metrics(
    y_test, hgb_tuned.predict_proba(X_test)[:, 1])["pr_auc"]
hgb = hgb_tuned if tuned_pr > default_pr else hgb_default
variant = "tuned" if tuned_pr > default_pr else "default"
print(f"Retrained HistGBM ({variant}) on train (n={len(X_train):,}) for SHAP")
print(f"  default test PR-AUC: {default_pr:.4f}")
print(f"  tuned   test PR-AUC: {tuned_pr:.4f}  -->  using {variant}")


# %% [markdown]
# ## B. Compute SHAP values (TreeExplainer)
#
# Sample 2,000 test users — SHAP over the full test set is unnecessary and slow. Global
# patterns show up cleanly at n=2000. Sanity-check that
# `sum(SHAP) + base_value == raw model margin`.

# %%
print("\n" + "=" * 70)
print("B. COMPUTE SHAP VALUES (TreeExplainer)")
print("=" * 70)
SAMPLE_SIZE = 2000
shap_values = compute_shap_values(hgb, X_test, sample_size=SAMPLE_SIZE)
X_shap = X_test.sample(n=SAMPLE_SIZE, random_state=42)
y_shap = y_test.loc[X_shap.index]
print(f"Computed SHAP for {SAMPLE_SIZE} test users")
print(f"Base value (expected log-odds output): {shap_values.base_values[0]:.4f}")
print(f"Base value (as probability): {1/(1+np.exp(-shap_values.base_values[0])):.4f}")
# Sanity check: sum(SHAP) + base_value == raw model output
# HistGBM's decision_function returns log-odds directly (equivalent to
# XGBoost's `output_margin=True`) — use that for the additivity check.
sample_pred = hgb.decision_function(X_shap.iloc[[0]])[0]
sample_check = shap_values.values[0].sum() + shap_values.base_values[0]
print(f"Sanity check (should match): raw margin={sample_pred:.4f}, "
      f"sum(shap)+base={sample_check:.4f}")


# %% [markdown]
# ## C. Global feature importance — mean |SHAP|
#
# Ranked feature contribution to the model's log-odds output — **magnitude only**.
# This chart answers "which features matter most?", not "does a high value push
# toward or away from churn?". For direction, see the beeswarm (Section D) or
# the dependence plots (Section E).
#
# **Why no color-by-direction here?** A signed-mean coloring mixes two things
# — the feature's magnitude and the sample's value distribution — and can flip
# the color for a feature whose HIGH values actually push AWAY from churn but
# whose LOW values (dominating a skewed sample) push toward churn. That's
# misleading. Keeping this chart to magnitude only, and reading direction off
# the beeswarm, avoids the trap.

# %%
print("\n" + "=" * 70)
print("C. GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
print("=" * 70)
importance = global_importance(shap_values, list(X_shap.columns), top_n=15)
print(importance.to_string(index=False))

# Bar chart -- single color, magnitude only. Direction lives in the beeswarm.
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1],
        color="#5B8FF9", alpha=0.85, edgecolor="white")
ax.set_xlabel("mean |SHAP value|  (avg contribution to log-odds)")
ax.set_title("Top 15 features by SHAP importance (magnitude)\n"
             "for direction of effect, see the beeswarm in Section D",
             fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_shap_global_importance.png",
            dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/05_shap_global_importance.png")
plt.show()


# %% [markdown]
# ## D. SHAP summary plot (beeswarm)
#
# One row per feature, one dot per user. X-position = SHAP contribution, color = feature
# value (red high, blue low). Shows **both magnitude and direction** of effect across
# the sample in one image.

# %%
print("\n" + "=" * 70)
print("D. SHAP SUMMARY PLOT (beeswarm)")
print("=" * 70)
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_shap, max_display=15, show=False,
                  plot_size=None)
plt.title("SHAP summary -- direction and magnitude of top features",
          fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_shap_beeswarm.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/05_shap_beeswarm.png")
plt.show()


# %% [markdown]
# ## E. Feature dependence for top 3 drivers
#
# Scatter of feature value (x) vs SHAP contribution (y) for the three most important
# features. Reveals thresholds and non-linearities the tree learned.

# %%
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


# %% [markdown]
# ## F. Local explanations — three worked examples
#
# Pick one user at each risk tier (highest, marginal, lowest) and walk through the top
# SHAP contributions. Attach the recommended intervention for the top `RISK+` feature.
# This is the pattern the Streamlit app uses in the Per-user lookup tab.

# %%
print("\n" + "=" * 70)
print("F. LOCAL EXPLANATIONS -- three worked examples")
print("=" * 70)

pred_proba = hgb.predict_proba(X_shap)[:, 1]

# "Marginal" = closest to the Phase 6 EV decision boundary, not an arbitrary
# probability like 0.15. Per-user break-even P* for each lever is
#     P*(churn) = cost / (uplift × LTV_user)
# A user becomes targetable when their P(churn) exceeds the CHEAPEST lever's
# P* (i.e., the min across levers -- first EV to cross zero). The marginal
# example is whoever sits closest to that personal boundary.
user_ltv = df.loc[X_shap.index, "plan_tier"].map(LTV_BY_TIER).values
per_lever_boundaries = np.array([
    params["cost"] / (params["uplift"] * user_ltv)
    for params in INTERVENTION_MENU.values()
])                                              # shape: (n_levers, n_users)
per_user_min_boundary = per_lever_boundaries.min(axis=0)
marginal_idx = int(np.argmin(np.abs(pred_proba - per_user_min_boundary)))

# Pick three example indices at different risk levels
# 1. Highest-risk user (model most confident of churn)
# 2. Marginal case at the user's personal EV break-even
# 3. Confidently retained user (control example)
example_indices = [
    ("HIGH RISK   ", int(np.argmax(pred_proba))),
    ("MARGINAL    ", marginal_idx),
    ("LOW RISK    ", int(np.argmin(pred_proba))),
]
print(f"Marginal user's EV break-even (cheapest lever): "
      f"P*={per_user_min_boundary[marginal_idx]:.3f}  |  "
      f"actual P(churn)={pred_proba[marginal_idx]:.3f}")

FEAT_W, VAL_W, SHAP_W = 32, 8, 7   # column widths for the compact table
for label, idx in example_indices:
    p = pred_proba[idx]
    actual = int(y_shap.iloc[idx])
    print(f"\n[{label.strip()}]  P(churn)={p:.3f}  actual={actual}")
    local = local_explanation(shap_values, X_shap, list(X_shap.columns),
                              idx=idx, top_n=5)
    # Compact fixed-width table -- avoids pandas' auto-truncation in Jupyter
    print(f"  {'feature':<{FEAT_W}} {'value':>{VAL_W}} {'shap':>{SHAP_W}}  dir")
    print(f"  {'-' * FEAT_W} {'-' * VAL_W} {'-' * SHAP_W}  ----")
    for _, r in local.iterrows():
        feat = str(r["feature"])[:FEAT_W]
        print(f"  {feat:<{FEAT_W}} "
              f"{float(r['feature_value']):>{VAL_W}.2f} "
              f"{r['shap_value']:>+{SHAP_W}.3f}  {r['direction']}")

    top_risk = local[local["direction"] == "RISK+"].head(1)
    if not top_risk.empty:
        feat = top_risk["feature"].iloc[0]
        lever = map_to_intervention(feat)
        tactical = lever.get("tactical_lever") or "none (diagnostic only)"
        print(f"  -> Diagnostic: {lever['lever']} -- {lever['note']}")
        print(f"     Tactical (Phase 6): {tactical}"
              + (f"  (cost=${lever['cost']:.0f})" if lever['cost'] > 0 else ""))


# %% [markdown]
# ## G. Diagnostic intervention map
#
# The bridge between model and decision rule. Every flagged feature has a
# paired **diagnostic lever** (or "no lever" if the feature is
# diagnostic-only) with a rich descriptive name that tells an analyst /
# PM reading SHAP output WHY the user is at risk. This mapping was
# curated with the Retention PM and lives in
# `src/models/explain.py:FEATURE_INTERVENTION_MAP`.
#
# Section H shows how these diagnostic categories collapse into Phase 6's
# 3 operational levers (`curated_playlist`, `credit_5`, `premium_upgrade`)
# — the actual crosswalk used by the retention platform.

# %%
print("\n" + "=" * 70)
print("G. DIAGNOSTIC INTERVENTION MAP")
print("=" * 70)
lever_rows = []
for feat, spec in FEATURE_INTERVENTION_MAP.items():
    lever_rows.append({
        "feature": feat,
        "lever":   spec["lever"],
        "cost":    spec["cost"],
        "note":    spec["note"],
    })
lever_df = pd.DataFrame(lever_rows)
# Hand-crafted fixed-width print to avoid pandas auto-truncation
FW, LW, CW = 32, 42, 6
print(f"  {'feature':<{FW}}  {'diagnostic_lever':<{LW}}  {'cost':>{CW}}  note")
print(f"  {'-'*FW}  {'-'*LW}  {'-'*CW}  ----")
for _, r in lever_df.iterrows():
    feat = str(r['feature'])[:FW]
    lev  = str(r['lever'])[:LW]
    cost = f"${r['cost']:.0f}" if r['cost'] > 0 else "  --"
    print(f"  {feat:<{FW}}  {lev:<{LW}}  {cost:>{CW}}  {r['note']}")


# %% [markdown]
# ## H. Diagnostic → tactical crosswalk (top 5 SHAP drivers)
#
# The rich diagnostic vocabulary in Section G collapses into Phase 6's
# **3 operational levers** through a crosswalk. Real retention platforms
# don't run 10 different automated campaigns — they run 2–4 well-tested
# ones with measured uplift. The diagnostic layer is for humans reading
# explanations; the tactical layer is what the platform actually sends.
#
# This section shows the crosswalk for the **top 5 SHAP drivers** from
# Section C — the features that actually move the model's output the most,
# so the ones Phase 6 will most often route on. Full crosswalk for all 35
# features lives in `src/models/explain.py:FEATURE_INTERVENTION_MAP`.
#
# **Rule of thumb:**
# - Engagement / content signals → `curated_playlist` (cheap, wide reach)
# - Support / payment / promo / trial-friction signals → `credit_5` (goodwill gesture)
# - Plan-structure / renewal signals → `premium_upgrade` (long-term retention)
# - Long-horizon or composite signals → None (diagnostic only)

# %%
print("\n" + "=" * 70)
print("H. DIAGNOSTIC -> TACTICAL CROSSWALK  (top 5 SHAP drivers)")
print("=" * 70)

top_5_features = importance.head(5)["feature"].tolist()

FW, DW, TW = 32, 42, 20
print(f"  {'feature':<{FW}}  {'diagnostic_lever':<{DW}}  {'tactical_lever':<{TW}}")
print(f"  {'-'*FW}  {'-'*DW}  {'-'*TW}")
for feat in top_5_features:
    spec = map_to_intervention(feat)
    diag = str(spec["lever"])[:DW]
    tact = str(spec.get("tactical_lever") or "-- diagnostic --")[:TW]
    print(f"  {str(feat)[:FW]:<{FW}}  {diag:<{DW}}  {tact:<{TW}}")

print(f"""
Read on the top 5 SHAP drivers:
  Each of the model's most influential features has an explicit route
  from diagnostic reason (what an analyst reads in SHAP) to tactical
  lever (what Phase 6 actually sends). No orphan diagnostic categories,
  no undocumented tactical assumptions. See Section G for the full
  35-feature diagnostic map and src/models/explain.py for the underlying
  crosswalk logic.
""")


# %% [markdown]
# ## I. Verdict + handoff to Phase 6
#
# Section H already shows the top 5 SHAP drivers with their diagnostic and
# tactical levers. This section rolls up the actionability count and
# describes what Phase 6 will consume from Phase 5.

# %%
print("\n" + "=" * 70)
print("I. PHASE 5 VERDICT + HANDOFF")
print("=" * 70)

# Actionability rollup across the top 15 SHAP drivers
top_15 = importance.head(15)["feature"].tolist()
n_actionable_top15 = sum(1 for f in top_15
                         if map_to_intervention(f).get("tactical_lever"))
print(f"\nActionability of the top 15 SHAP drivers:")
print(f"  * {n_actionable_top15} / 15 route to a tactical lever "
      f"(curated_playlist, credit_5, or premium_upgrade)")
print(f"  * {15 - n_actionable_top15} / 15 are diagnostic-only "
      f"(structural / long-horizon / composite)")

print("\nHandoff to Phase 6 (cost-aware decision rule):")
print("  - Per-user P(churn) from the calibrated HistGBM")
print("  - Per-user top SHAP driver + its tactical lever (Section H crosswalk)")
print("  - Intervention cost + expected uplift from INTERVENTION_MENU")
print("  - LTV by plan tier from the scenario brief")
print("\nDecision rule = argmax over levers of (P(churn) x uplift x LTV - cost).")
