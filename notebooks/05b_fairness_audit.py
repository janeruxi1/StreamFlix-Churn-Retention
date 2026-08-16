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
# # Phase 5b — Fairness & Segment Performance Audit
#
# **Scope.** This notebook audits the **v1 model** (calibrated HistGradientBoosting
# from Phase 4) and the **v1 policy** (cost-aware EV rule from Phase 6). The v2
# uplift model (Phase 8) and its head-to-head (Phase 9) are out of scope — a
# separate fairness audit would be needed if v2 ships to production, and the
# right metrics for a causal-uplift policy differ (e.g., differential ITE
# estimation quality by segment, not just discrimination + calibration parity).
#
# **Why this exists.** A model can have great aggregate PR-AUC and still fail
# systematically on a subgroup. Before shipping v1 (or v2) to the retention team, we
# need to verify:
#
# 1. **Discrimination is comparable across segments** — PR-AUC by plan tier, tenure
#    bucket, country, engagement cohort
# 2. **Probabilities mean the same thing across segments** — calibration parity. If
#    `P(churn) = 0.20` maps to 25% actual churn for Basic users but 15% for Premium,
#    the decision-rule budget is being spent unequally
# 3. **The targeted policy doesn't over- or under-serve any group** — targeting rate
#    parity. If Premium users get 5× the intervention offers of Basic users, that's a
#    business-relevant asymmetry (positive or negative — depends on strategy)
# 4. **Recall parity on true churners** — among users who actually churn, does the
#    model flag Basic users at the same rate as Premium?
#
# ## Sections
#
# | Section | Check |
# |---|---|
# | **A. Setup** | Score all subscribers with the v1 churn model |
# | **B. PR-AUC by segment** | Discrimination parity across plan tier, tenure, country |
# | **C. Calibration by segment** | Reliability curves stratified by plan tier |
# | **D. Brier by segment** | Combined discrimination + calibration score |
# | **E. Targeting rate by segment** | Does the policy over/under-serve any group? |
# | **F. Recall parity (equal opportunity)** | Among true churners, who gets flagged? |
# | **G. Verdict** | Which segments are OK, which need attention |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
from pathlib import Path

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
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import prepare_features
from src.models.production import load_production_churn_model
from src.models.evaluate import calibration_curve_points
from src.decisions.policy import (
    INTERVENTION_MENU, LTV_BY_TIER,
    pick_best_lever, apply_budget_cap, apply_premium_cap,
    PREMIUM_UPGRADE_CAP_PCT,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
BUDGET = 200_000.0


# %% [markdown]
# ## A. Setup — score all subscribers with the v1 churn model

# %%
print("=" * 70)
print("A. SETUP")
print("=" * 70)

raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)
X, y = prepare_features(df)

model, artifact = load_production_churn_model()
X = X[artifact["feature_names"]]
p_churn = model.predict_proba(X)[:, 1]
y_true = y.values

# Segments to audit -- pull straight from raw features, not one-hots
segments = pd.DataFrame({
    "plan_tier":         raw["plan_tier"].values,
    "billing_cycle":     raw["billing_cycle"].values,
    "country":           raw["country"].values,
    "engagement_cohort": raw["engagement_cohort"].values,
    "tenure_bucket": pd.cut(
        raw["tenure_months"],
        bins=[-1, 2, 5, 11, 12, 24, 60],
        labels=["m0-2 trial", "m3-5", "m6-11", "m12", "m13-24", "m25+"],
    ),
})

print(f"Scored {len(df):,} subscribers")
print(f"Base rate churn: {y_true.mean():.2%}")


# %% [markdown]
# ## B. Discrimination by segment — PR-AUC parity
#
# For each segment value we compute PR-AUC on that subset. If the model is fair, all
# subsets should score within a reasonable band of the overall (accounting for sample
# size and base-rate differences).

# %%
print("\n" + "=" * 70)
print("B. PR-AUC BY SEGMENT")
print("=" * 70)

overall_pr = average_precision_score(y_true, p_churn)
overall_roc = roc_auc_score(y_true, p_churn)
overall_brier = brier_score_loss(y_true, p_churn)
print(f"\nOverall:  PR-AUC = {overall_pr:.4f}  ROC-AUC = {overall_roc:.4f}  "
      f"Brier = {overall_brier:.4f}\n")


def per_segment_metrics(seg_col: str) -> pd.DataFrame:
    rows = []
    for val, sub in segments.groupby(seg_col, observed=True):
        idx = sub.index
        y_sub = y_true[idx]
        p_sub = p_churn[idx]
        n = len(sub)
        pos = int(y_sub.sum())
        if pos < 5 or n - pos < 5:
            # Too few of either class to compute metrics reliably
            rows.append({"segment": str(val), "n": n, "pos_rate": pos / n,
                         "pr_auc": np.nan, "roc_auc": np.nan,
                         "brier": np.nan})
            continue
        rows.append({
            "segment":  str(val),
            "n":        n,
            "pos_rate": pos / n,
            "pr_auc":   average_precision_score(y_sub, p_sub),
            "roc_auc":  roc_auc_score(y_sub, p_sub),
            "brier":    brier_score_loss(y_sub, p_sub),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


for seg_col in ["plan_tier", "billing_cycle", "country",
                "engagement_cohort", "tenure_bucket"]:
    print(f"\nBy {seg_col}:")
    print(per_segment_metrics(seg_col).round(4).to_string(index=False))


# %% [markdown]
# ## C. Calibration parity by plan tier
#
# Does `P(churn) = 0.20` mean the same thing for a Basic user as for a Premium user?
# If the reliability curves for the three tiers diverge, the decision rule's EV math is
# systematically off for some tiers -- budget gets over- or under-spent on them.

# %%
print("\n" + "=" * 70)
print("C. CALIBRATION PARITY BY PLAN TIER")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 7))
colors = {"Basic": "#F6735B", "Standard": "#5B8FF9", "Premium": "#5AD8A6"}
for tier, color in colors.items():
    mask = segments["plan_tier"] == tier
    y_sub = y_true[mask]
    p_sub = p_churn[mask]
    if len(y_sub) < 100:
        continue
    cal = calibration_curve_points(y_sub, p_sub, n_bins=8)
    ax.plot(cal["mean_pred"], cal["frac_positive"], marker="o",
            linewidth=2, color=color,
            label=f"{tier} (n={len(y_sub):,}, pos={y_sub.mean():.1%})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
        linewidth=1, label="perfectly calibrated")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("actual fraction positive")
ax.set_title("Calibration parity by plan tier\n"
             "(overlapping curves = model is fair across tiers)",
             fontweight="bold")
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.6)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_calibration_by_tier.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/10_calibration_by_tier.png")
plt.show()


# %% [markdown]
# ## C.2. Calibration parity by engagement cohort
#
# The interview prep + memo reference a **calibration gap on the casual cohort**.
# The plan-tier calibration in Section C is standard; this stratifies by
# engagement cohort, which is where the real story lives — engagement patterns
# drive churn behavior more than plan tier does, so calibration miscalibration
# is more likely to show up here.

# %%
print("\n" + "=" * 70)
print("C.2. CALIBRATION PARITY BY ENGAGEMENT COHORT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 7))
cohort_colors = {
    "casual":  "#F6735B",
    "regular": "#5B8FF9",
    "power":   "#5AD8A6",
}
for cohort in segments["engagement_cohort"].unique():
    mask = segments["engagement_cohort"] == cohort
    y_sub = y_true[mask]
    p_sub = p_churn[mask]
    if len(y_sub) < 100:
        continue
    cal = calibration_curve_points(y_sub, p_sub, n_bins=8)
    color = cohort_colors.get(str(cohort), "gray")
    ax.plot(cal["mean_pred"], cal["frac_positive"], marker="o",
            linewidth=2, color=color,
            label=f"{cohort} (n={len(y_sub):,}, pos={y_sub.mean():.1%})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
        linewidth=1, label="perfectly calibrated")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("actual fraction positive")
ax.set_title("Calibration parity by engagement cohort\n"
             "(divergence from the diagonal = miscalibration for that cohort)",
             fontweight="bold")
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.6)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_calibration_by_cohort.png",
            dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/10_calibration_by_cohort.png")
plt.show()


# %% [markdown]
# ## D. Brier score by segment (combined discrimination + calibration)
#
# Brier score combines discrimination AND calibration in one number. Lower is better.
# Segment-level Brier close to the overall Brier = model works equally well on that
# segment. Big gaps = red flags.

# %%
print("\n" + "=" * 70)
print("D. BRIER SCORE BY SEGMENT")
print("=" * 70)

brier_rows = []
for seg_col in ["plan_tier", "engagement_cohort", "tenure_bucket"]:
    for val, sub in segments.groupby(seg_col, observed=True):
        idx = sub.index
        y_sub = y_true[idx]
        p_sub = p_churn[idx]
        if len(y_sub) < 100:
            continue
        b = brier_score_loss(y_sub, p_sub)
        brier_rows.append({
            "segment_col": seg_col, "segment": str(val),
            "n": len(y_sub), "brier": b,
            "delta_vs_overall": b - overall_brier,
        })
brier_df = pd.DataFrame(brier_rows).sort_values("delta_vs_overall")
print(brier_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
brier_df_sorted = brier_df.sort_values("delta_vs_overall")
bar_colors = [
    "#5AD8A6" if abs(d) < 0.005
    else ("#F6BD16" if abs(d) < 0.01 else "#F6735B")
    for d in brier_df_sorted["delta_vs_overall"]
]
ax.barh([f"{r['segment_col']}={r['segment']}" for _, r in brier_df_sorted.iterrows()],
        brier_df_sorted["delta_vs_overall"],
        color=bar_colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.6)
ax.axvspan(-0.005, 0.005, alpha=0.08, color="green",
           label="within noise (< 0.005)")
ax.set_xlabel("Brier delta vs overall  (positive = worse than overall)")
ax.set_title("Brier score parity across segments\n"
             "green bars = within noise band; red = notable gap",
             fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_brier_by_segment.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/10_brier_by_segment.png")
plt.show()


# %% [markdown]
# ## E. Targeting rate parity — does the v1 policy over/under-serve any group?
#
# Run v1 policy (propensity, full menu, $200k budget) and check what fraction of each
# segment gets targeted. A big spread means the policy is concentrated on certain
# segments -- may be legitimate (highest-EV users are in that segment) or may be a
# fairness concern (segment gets systematically ignored).

# %%
print("\n" + "=" * 70)
print("E. TARGETING RATE BY SEGMENT (v1 policy)")
print("=" * 70)

ltv = raw["plan_tier"].map(LTV_BY_TIER).values
policy = pick_best_lever(p_churn, ltv, INTERVENTION_MENU)
policy = apply_budget_cap(policy, budget=BUDGET)
policy = apply_premium_cap(policy, n_total=len(policy),
                           cap_pct=PREMIUM_UPGRADE_CAP_PCT)
targeted = policy["will_target"].values

overall_target_rate = targeted.mean()
print(f"\nOverall targeting rate: {overall_target_rate:.2%}")
print(f"(This is the fraction of the {len(policy):,}-user base the policy "
      f"targets under the ${BUDGET/1000:.0f}k governance ceiling. Segment "
      f"target rates below are stated relative to this baseline.)\n")

# Collect per-segment target rates so Section G can auto-flag any that
# are systematically under-served (rate_ratio_vs_overall < 0.5).
target_rate_rows = []
for seg_col in ["plan_tier", "engagement_cohort", "tenure_bucket", "country"]:
    print(f"By {seg_col}:")
    seg_rows = []
    for val, sub in segments.groupby(seg_col, observed=True):
        idx = sub.index
        n = len(sub)
        n_targeted = int(targeted[idx].sum())
        rate = n_targeted / n if n else 0
        row = {
            "segment_col": seg_col, "segment": str(val),
            "n": n, "n_targeted": n_targeted,
            "target_rate": rate,
            "rate_ratio_vs_overall": rate / overall_target_rate
            if overall_target_rate > 0 else 0,
        }
        seg_rows.append(row)
        target_rate_rows.append(row)
    print(pd.DataFrame(seg_rows).drop(columns=["segment_col"])
          .round(3).to_string(index=False))
    print()
target_rate_df = pd.DataFrame(target_rate_rows)


# %% [markdown]
# ## F. Recall parity (equal opportunity) among true churners
#
# Of the users who actually churned, what fraction did the policy target?
# If the recall is 40% for one segment and 15% for another, we're systematically
# leaving churners on the table for the under-served group. This is the classic
# "equal opportunity" fairness metric.

# %%
print("\n" + "=" * 70)
print("F. RECALL PARITY ON TRUE CHURNERS (v1 policy)")
print("=" * 70)

churner_mask = y_true == 1
overall_recall = targeted[churner_mask].mean() if churner_mask.sum() else 0
print(f"\nOverall recall on true churners: {overall_recall:.2%}\n")

# Collect per-segment recall so Section G can auto-flag any segment with
# recall_ratio_vs_overall < 0.5 (classic equal-opportunity failure).
recall_rows = []
for seg_col in ["plan_tier", "engagement_cohort", "tenure_bucket"]:
    print(f"By {seg_col}:")
    rows = []
    for val, sub in segments.groupby(seg_col, observed=True):
        idx = sub.index
        y_sub = y_true[idx]
        churners = idx[y_sub == 1]
        n_churners = len(churners)
        if n_churners == 0:
            continue
        recall = targeted[churners].mean()
        row = {
            "segment_col": seg_col, "segment": str(val),
            "n_churners": n_churners,
            "recall": recall,
            "recall_ratio_vs_overall": recall / overall_recall
            if overall_recall else 0,
        }
        rows.append(row)
        recall_rows.append(row)
    print(pd.DataFrame(rows).drop(columns=["segment_col"])
          .round(3).to_string(index=False))
    print()
recall_df = pd.DataFrame(recall_rows)


# %% [markdown]
# ## G. Verdict — segment audit summary

# %%
print("\n" + "=" * 70)
print("G. FAIRNESS AUDIT VERDICT")
print("=" * 70)

# ------------------------------------------------------------------
# Thresholds for auto-flagging concerns.
#
# BRIER_DELTA_THRESHOLD = 0.01
#   Rule of thumb: Brier is bounded [0, 1] and our overall Brier is
#   ~0.047 for this dataset. A 0.01 absolute shift is ~20% relative --
#   above measurement noise, below "catastrophically miscalibrated".
#
# RATE_RATIO_THRESHOLD = 0.5
#   Standard "80% rule"-adjacent heuristic (we use 0.5 = 2x asymmetry
#   as the harder line). Segments getting < half the overall targeting
#   or recall rate are systematically under-served.
# ------------------------------------------------------------------
BRIER_DELTA_THRESHOLD = 0.01
RATE_RATIO_THRESHOLD = 0.5

concerns = []

# --- (a) Calibration / Brier concerns -----------------------------
for _, row in brier_df.iterrows():
    if row["delta_vs_overall"] > BRIER_DELTA_THRESHOLD:
        concerns.append({
            "kind": "BRIER",
            "segment": f"{row['segment_col']}={row['segment']}",
            "detail": (f"Brier {row['brier']:.4f} vs overall "
                       f"{overall_brier:.4f} (Δ +{row['delta_vs_overall']:.4f})"),
            "action": ("Recommend a segment-specific Platt calibrator, or "
                       "retrain with segment-stratified calibration set."),
        })

# --- (b) Under-service in targeting rate --------------------------
for _, row in target_rate_df.iterrows():
    if row["rate_ratio_vs_overall"] < RATE_RATIO_THRESHOLD:
        concerns.append({
            "kind": "UNDER-SERVED",
            "segment": f"{row['segment_col']}={row['segment']}",
            "detail": (f"target rate {row['target_rate']:.2%} = "
                       f"{row['rate_ratio_vs_overall']:.2f}x overall "
                       f"({overall_target_rate:.2%})"),
            "action": ("Check for feature drift or LTV under-estimation for "
                       "this segment; may need cohort-specific EV thresholds."),
        })

# --- (c) Equal-opportunity failure in recall ----------------------
for _, row in recall_df.iterrows():
    if row["recall_ratio_vs_overall"] < RATE_RATIO_THRESHOLD:
        concerns.append({
            "kind": "RECALL",
            "segment": f"{row['segment_col']}={row['segment']}",
            "detail": (f"recall on true churners {row['recall']:.2%} = "
                       f"{row['recall_ratio_vs_overall']:.2f}x overall "
                       f"({overall_recall:.2%})"),
            "action": ("Recommend cohort-weighted retraining, or a "
                       "segment-specific score adjustment to raise recall "
                       "on this group."),
        })

print(f"""
Summary:
  Overall PR-AUC:                {overall_pr:.4f}
  Overall Brier:                 {overall_brier:.4f}
  Overall v1 targeting rate:     {overall_target_rate:.2%}
  Overall v1 recall on churners: {overall_recall:.2%}

Thresholds used to auto-flag concerns:
  Brier delta > {BRIER_DELTA_THRESHOLD} (~20% relative to overall)
  Target-rate ratio  < {RATE_RATIO_THRESHOLD} (segment under-served)
  Recall ratio       < {RATE_RATIO_THRESHOLD} (equal-opportunity failure)

Segment-level concerns detected: {len(concerns)}
""")

for c in concerns:
    print(f"  [{c['kind']:<12}] {c['segment']}")
    print(f"                  {c['detail']}")
    print(f"                  -> {c['action']}\n")

if not concerns:
    print(f"""  None. All segments perform within noise on all three checks:
    Brier delta < {BRIER_DELTA_THRESHOLD}, target rate >= {RATE_RATIO_THRESHOLD}x overall,
    recall >= {RATE_RATIO_THRESHOLD}x overall. Targeting-rate variation across segments
    reflects legitimate churn-rate + LTV differences captured by the
    model, not systematic under-service.
""")

print("""
Reading this audit for the memo:
  - Calibration is the key thing to watch. If P(churn) means different
    things for different tiers / cohorts, the EV budget is being spent
    unequally even when the aggregate numbers look fine.
  - Targeting-rate spread across segments is EXPECTED (higher-churn or
    higher-LTV segments legitimately get more intervention). What's
    NOT expected is a segment being systematically dropped -- caught
    by the UNDER-SERVED check above.
  - Recall parity is the equal-opportunity metric: does the policy
    catch churners at the same rate across segments? A big spread
    signals systematic under-service -- caught by the RECALL check.

Charts for the memo:
  - reports/figures/10_calibration_by_tier.png
  - reports/figures/10_calibration_by_cohort.png    (casual-cohort story)
  - reports/figures/10_brier_by_segment.png
""")
