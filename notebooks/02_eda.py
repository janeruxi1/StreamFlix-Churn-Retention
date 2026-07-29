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
# # Phase 2 — EDA + Survival Analysis
#
# With the data validated in Phase 1, we now look for the **patterns** the model will
# need to capture and the **segments** that matter for the cost-aware decision rule
# downstream.
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Population overview** | Who is in the dataset |
# | **B. Churn rates by segment** | Which categorical slices differ from baseline |
# | **C. Kaplan-Meier survival** | Structural covariates (5 strata) |
# | **C2. Landmark analysis** | Handles time-varying covariate (engagement_cohort) |
# | **C3. Sensitivity check** | Landmark-choice robustness |
# | **D. Hazard by tenure** | m2 / m12 spike visualization |
# | **E. Engagement trend** | Churners vs non-churners distribution |
# | **F. Tenure × cohort heatmap** | 2D segment view |
# | **G. Verdict** | Handoff to Phase 3 (feature engineering) |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
from pathlib import Path

# Run from project root whether invoked as `python notebooks/02_...` or
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

from src.data.loader import load_subscribers

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load_subscribers("data/subscribers.csv")


# %% [markdown]
# ## A. Population overview
#
# Shares of every categorical/boolean segment — sanity check that the dataset composition
# matches what the PM brief describes.

# %%
print("=" * 70)
print("A. POPULATION OVERVIEW")
print("=" * 70)
for col in ["plan_tier", "billing_cycle", "country", "payment_method",
            "engagement_cohort", "auto_renew", "multi_profile", "promo_active"]:
    if df[col].dtype == bool:
        share = df[col].mean()
        print(f"  {col:<22}  True share = {share:.1%}")
    else:
        top_shares = df[col].value_counts(normalize=True).round(3)
        print(f"  {col}:")
        for k, v in top_shares.items():
            print(f"      {k:<15} {v:.1%}")


# %% [markdown]
# ## B. Churn rate by categorical segment
#
# Which segments churn at ≥ 1.5× the baseline rate. These are the natural candidates for
# targeted intervention in the Phase 6 policy.

# %%
print("\n" + "=" * 70)
print("B. CHURN RATE BY CATEGORICAL SEGMENT")
print("=" * 70)

seg_cols = ["plan_tier", "billing_cycle", "country", "payment_method",
            "engagement_cohort", "auto_renew", "multi_profile"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
overall_rate = df["churned_next_30d"].mean()

for ax, col in zip(axes, seg_cols):
    seg = df.groupby(col)["churned_next_30d"].agg(["mean", "count"]).reset_index()
    seg = seg.sort_values("mean", ascending=False)
    bars = ax.bar(seg[col].astype(str), seg["mean"] * 100,
                  color="#5B8FF9", alpha=0.85, edgecolor="white")
    ax.axhline(overall_rate * 100, color="#F6AD55", linestyle="--",
               linewidth=1.2, label=f"overall {overall_rate:.1%}")
    for b, rate, n in zip(bars, seg["mean"], seg["count"]):
        ax.text(b.get_x() + b.get_width() / 2,
                rate * 100 + 0.1,
                f"{rate:.1%}\n(n={n:,})",
                ha="center", fontsize=8)
    ax.set_title(col, fontweight="bold", fontsize=10)
    ax.set_ylabel("churn rate (%)", fontsize=8)
    ax.set_ylim(0, max(seg["mean"]) * 100 * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_visible(False)
plt.suptitle("Churn rate by segment (vs overall baseline)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_churn_by_segment.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_churn_by_segment.png")

print(f"\nOverall churn rate: {overall_rate:.2%}")
print("Segments with churn rate >= 1.5x baseline:")
for col in seg_cols:
    seg = df.groupby(col)["churned_next_30d"].mean()
    for seg_val, rate in seg.items():
        if rate >= 1.5 * overall_rate:
            print(f"  {col:<20} = {seg_val:<15}  rate = {rate:.2%}")


# %% [markdown]
# ## C. Kaplan-Meier survival curves — structural covariates
#
# Stratify on covariates set **at or near signup** that stay stable (plan_tier,
# billing_cycle, payment_method, auto_renew). KM's *"covariate fixed at t=0"* assumption
# holds for these. Time-varying covariates (engagement_cohort) are handled separately in
# C2 with landmark analysis.

# %%
print("\n" + "=" * 70)
print("C. KAPLAN-MEIER SURVIVAL CURVES (structural covariates)")
print("=" * 70)


def km(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Simple Kaplan-Meier estimator -- no lifelines dependency.

    durations : tenure_months at observation
    events    : 1 if churned in the next 30 days, 0 if censored (still active)
    """
    durations = np.asarray(durations)
    events = np.asarray(events)
    event_times = np.sort(np.unique(durations[events == 1]))
    rows = []
    s = 1.0
    for t in event_times:
        d = ((durations == t) & (events == 1)).sum()
        n = (durations >= t).sum()
        if n == 0:
            break
        s *= (1 - d / n)
        rows.append((t, s, n, d))
    return pd.DataFrame(rows, columns=["t", "S", "at_risk", "events"])


# Single 5-panel figure: overall + 4 structural covariates
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# Panel 1: overall
ax = axes[0]
overall = km(df["tenure_months"], df["churned_next_30d"])
ax.step(overall["t"], overall["S"], where="post", color="#5B8FF9", linewidth=2)
ax.set_title("Overall survival S(t)", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("P(still subscribed)")
ax.set_ylim(0.5, 1.01)
ax.grid(True, linestyle="--", alpha=0.4)

# Panel 2: plan tier
ax = axes[1]
for plan, color in {"Basic": "#F6735B", "Standard": "#F6BD16",
                    "Premium": "#5AD8A6"}.items():
    sub = df[df["plan_tier"] == plan]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{plan} (n={len(sub):,})")
ax.set_title("Survival by plan tier", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Panel 3: billing cycle
ax = axes[2]
for billing, color in {"monthly": "#F6735B", "annual": "#5AD8A6"}.items():
    sub = df[df["billing_cycle"] == billing]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{billing} (n={len(sub):,})")
ax.set_title("Survival by billing cycle", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Panel 4: auto-renew
ax = axes[3]
for ar, color, label in [(True, "#5AD8A6", "auto-renew ON"),
                          (False, "#F6735B", "auto-renew OFF")]:
    sub = df[df["auto_renew"] == ar]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{label} (n={len(sub):,})")
ax.set_title("Survival by auto-renew status", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("P(still subscribed)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Panel 5: payment method
ax = axes[4]
pm_colors = {"credit_card": "#5AD8A6", "paypal": "#5B8FF9",
             "gift_card": "#F6735B"}
for pm, color in pm_colors.items():
    sub = df[df["payment_method"] == pm]
    if len(sub) == 0:
        continue
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{pm} (n={len(sub):,})")
ax.set_title("Survival by payment method", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Hide the 6th (empty) panel
axes[5].set_visible(False)

plt.suptitle("Kaplan-Meier survival curves -- structural covariates",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_kaplan_meier.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_kaplan_meier.png")

# Numerical readout for auto_renew and payment_method (smallest n cohorts)
for col in ["auto_renew", "payment_method"]:
    print(f"\nChurn rate by {col}:")
    print(df.groupby(col)["churned_next_30d"].agg(["mean", "count"]).round(4))


# %% [markdown]
# ## C2. Landmark analysis — engagement_cohort (time-varying covariate)
#
# `engagement_cohort` is **time-varying**: a user labeled `casual` at month 3 might be
# `heavy` at month 12. Naive KM stratification by current cohort suffers from
# **immortal-time bias** — the `heavy` cohort is enriched with users who survived long
# enough to *become* heavy.
#
# Landmark analysis fixes this:
# 1. Condition on survival to a landmark time `t*`
# 2. Stratify by covariate value at `t*`
# 3. Compute survival **forward** from `t*`
#
# In our cross-sectional snapshot we observe cohort once, so the landmark approximation
# is *"restrict to users who survived to t*=6"* — users whose cohort label is more stable
# and less affected by survival selection.

# %%
print("\n" + "=" * 70)
print("C2. LANDMARK ANALYSIS -- engagement_cohort")
print("=" * 70)

LANDMARK = 6
established = df[df["tenure_months"] >= LANDMARK].copy()
print(f"\nLandmark t* = {LANDMARK} months")
print(f"Full sample:        n={len(df):,}")
print(f"Landmark-restricted: n={len(established):,} "
      f"({len(established)/len(df):.1%})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cohort_colors = {"heavy": "#5AD8A6", "regular": "#5B8FF9",
                 "casual": "#F6735B"}

ax = axes[0]
for cohort, color in cohort_colors.items():
    sub = df[df["engagement_cohort"] == cohort]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{cohort} (n={len(sub):,})")
ax.set_title("Naive KM by current cohort\n"
             "(biased -- cohort label is time-varying)",
             fontweight="bold", fontsize=10)
ax.set_xlabel("tenure (months)")
ax.set_ylabel("P(still subscribed)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

ax = axes[1]
for cohort, color in cohort_colors.items():
    sub = established[established["engagement_cohort"] == cohort]
    if len(sub) == 0:
        continue
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{cohort} (n={len(sub):,})")
ax.axvline(LANDMARK, color="black", linestyle=":", linewidth=1.5,
           alpha=0.6, label=f"landmark t*={LANDMARK}")
ax.set_title(f"Landmark KM by cohort\n"
             f"(conditioned on survival to month {LANDMARK})",
             fontweight="bold", fontsize=10)
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Engagement cohort survival -- naive vs landmark "
             "(handling time-varying covariates)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_km_cohort_landmark.png",
            dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_km_cohort_landmark.png")

print("\nChurn-rate comparison: naive (all users) vs landmark (tenure>=6):")
naive = df.groupby("engagement_cohort")["churned_next_30d"].mean()
landmark = established.groupby("engagement_cohort")["churned_next_30d"].mean()
comp = pd.DataFrame({"naive": naive, "landmark": landmark}).round(4) * 100
comp["delta_pp"] = (comp["landmark"] - comp["naive"]).round(2)
print(comp)


# %% [markdown]
# ## C3. Sensitivity check — does the cohort gap survive across `t*` choices?
#
# A single landmark choice is a judgment call. Sweep across plausible values (2, 4, 6, 9,
# 12) and check whether the cohort gap survives at every choice. If it does, the finding
# is robust; if it flips, the analysis is fragile.

# %%
print("\n" + "=" * 70)
print("C3. SENSITIVITY CHECK -- landmark robustness")
print("=" * 70)

landmark_values = [2, 4, 6, 9, 12]
rows = []
for t_star in landmark_values:
    sub = df[df["tenure_months"] >= t_star]
    n_kept = len(sub)
    pct_kept = n_kept / len(df)
    by_cohort = sub.groupby("engagement_cohort")["churned_next_30d"].mean()
    casual_rate = by_cohort.get("casual", np.nan)
    heavy_rate = by_cohort.get("heavy", np.nan)
    gap_pp = (casual_rate - heavy_rate) * 100
    rows.append({
        "t_star": t_star,
        "n_kept": n_kept,
        "pct_kept": round(pct_kept * 100, 1),
        "casual_pct": round(casual_rate * 100, 2),
        "heavy_pct": round(heavy_rate * 100, 2),
        "gap_pp": round(gap_pp, 2),
    })
sensitivity = pd.DataFrame(rows)
print("\nCasual-vs-heavy churn gap across landmark choices:")
print(sensitivity.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 5))

color_gap = "#5B8FF9"
ax1.plot(sensitivity["t_star"], sensitivity["gap_pp"],
         marker="o", markersize=10, linewidth=2.5, color=color_gap,
         label="casual - heavy gap (pp)", zorder=3)
ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
ax1.axvline(6, color="#F6AD55", linestyle="--", linewidth=1.5,
            alpha=0.7, label="chosen t*=6")
ax1.set_xlabel("landmark t* (months)")
ax1.set_ylabel("casual - heavy churn gap (pp)", color=color_gap)
ax1.tick_params(axis="y", labelcolor=color_gap)
ax1.set_ylim(0, sensitivity["gap_pp"].max() * 1.25)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
color_size = "#F6735B"
ax2.bar(sensitivity["t_star"], sensitivity["pct_kept"], alpha=0.25,
        color=color_size, width=0.8, label="% sample retained", zorder=1)
ax2.set_ylabel("% sample retained", color=color_size)
ax2.tick_params(axis="y", labelcolor=color_size)
ax2.set_ylim(0, 100)

plt.title("Landmark sensitivity: cohort gap is robust; sample shrinks fast",
          fontweight="bold")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_landmark_sensitivity.png",
            dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/02_landmark_sensitivity.png")


# %% [markdown]
# ## D. Hazard rate by tenure month
#
# Verify the two spikes the PM brief calls out: **m2 (post-trial drop)** and
# **m12 (annual reassessment)**. These are the timing windows for the retention team's
# blanket-m11 campaign that Phase 6 will beat.

# %%
print("\n" + "=" * 70)
print("D. HAZARD RATE BY TENURE MONTH (m2/m12 spike check)")
print("=" * 70)

tenure_hazard = (
    df.groupby("tenure_months")["churned_next_30d"]
      .agg(["mean", "count"])
      .reset_index()
      .rename(columns={"mean": "hazard"})
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(tenure_hazard["tenure_months"], tenure_hazard["hazard"] * 100,
       color="#5B8FF9", alpha=0.85, edgecolor="white", width=0.85)
ax.axhline(overall_rate * 100, color="gray", linestyle="--",
           linewidth=1, label=f"overall {overall_rate:.1%}")
ax.axvline(2, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m2 trial drop")
ax.axvline(12, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m12 anniversary")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("hazard = 30-day churn rate (%)")
ax.set_title("Hazard rate by tenure month -- m2 and m12 spikes",
             fontweight="bold")
ax.set_xlim(-0.5, 36)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_tenure_hazard.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_tenure_hazard.png")


# %% [markdown]
# ## E. Engagement-trend distribution — churners vs non-churners
#
# `watch_trend_7d_to_30d = watch_hours_7d × (30/7) / watch_hours_30d`. Values `< 1` mean
# a user is watching *less* in the recent week than their monthly baseline (declining
# engagement); `> 1` means they're heating up. Churners should skew heavily below 1.

# %%
print("\n" + "=" * 70)
print("E. ENGAGEMENT TREND -- churners vs non-churners")
print("=" * 70)

df["watch_trend_7d_to_30d"] = (
    df["watch_hours_last_7d"] * (30 / 7) /
    df["watch_hours_last_30d"].clip(lower=0.1)
).clip(upper=3.0)

print("\nTrend ratio statistics (watch_hours_7d x 30/7 / watch_hours_30d):")
print(df.groupby("churned_next_30d")["watch_trend_7d_to_30d"]
        .describe(percentiles=[0.25, 0.5, 0.75]).round(3))

fig, ax = plt.subplots(figsize=(10, 5))
for outcome, color, label in [
    (0, "#5AD8A6", "did NOT churn"),
    (1, "#F6735B", "churned"),
]:
    vals = df.loc[df["churned_next_30d"] == outcome, "watch_trend_7d_to_30d"]
    ax.hist(vals, bins=50, alpha=0.55, label=f"{label} (n={len(vals):,})",
            color=color, edgecolor="white", density=True)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="flat (=1)")
ax.set_xlabel("watch_trend_7d_to_30d  (<1 = declining, >1 = growing)")
ax.set_ylabel("density")
ax.set_title("Engagement trend by churn outcome", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_engagement_trend.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_engagement_trend.png")


# %% [markdown]
# ## F. Tenure × engagement-cohort heatmap
#
# 2D view: how does churn rate vary across the interaction of tenure bucket and
# engagement cohort? The extremes of this heatmap are where targeted intervention would
# have the biggest per-user ROI.

# %%
print("\n" + "=" * 70)
print("F. CHURN HEATMAP: tenure bucket x engagement cohort")
print("=" * 70)

df["tenure_bucket"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 1, 2, 5, 11, 13, 24, 60],
    labels=["m0-1", "m2 (trial)", "m3-5", "m6-11", "m12-13 (anniv)",
            "m14-24", "m25+"],
)
heatmap = df.pivot_table(
    index="tenure_bucket", columns="engagement_cohort",
    values="churned_next_30d", aggfunc="mean", observed=True,
)
cohort_order = ["heavy", "regular", "casual"]
heatmap = heatmap[cohort_order]

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heatmap.values * 100, cmap="RdYlGn_r", aspect="auto",
               vmin=0, vmax=heatmap.values.max() * 100)
ax.set_xticks(range(len(heatmap.columns)))
ax.set_xticklabels(heatmap.columns)
ax.set_yticks(range(len(heatmap.index)))
ax.set_yticklabels(heatmap.index)
ax.set_xlabel("engagement cohort")
ax.set_ylabel("tenure bucket")
ax.set_title("Churn rate (%) by tenure x engagement cohort",
             fontweight="bold")
for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        ax.text(j, i, f"{heatmap.values[i, j] * 100:.1f}%",
                ha="center", va="center", fontsize=10,
                color="white" if heatmap.values[i, j] * 100 > 6 else "black")
plt.colorbar(im, ax=ax, label="churn rate (%)")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_tenure_cohort_heatmap.png",
            dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_tenure_cohort_heatmap.png")
print(f"\nHeatmap values:\n{(heatmap * 100).round(2)}")


# %% [markdown]
# ## G. Verdict — handoff to Phase 3
#
# Distills the key patterns Phase 3 needs to encode as features: plan-tier spread,
# engagement-trend divergence between churners and non-churners, and the tenure × cohort
# gap.

# %%
print("\n" + "=" * 70)
print("G. PHASE 2 VERDICT")
print("=" * 70)

trend_decline = (df.loc[df["churned_next_30d"] == 1, "watch_trend_7d_to_30d"]
                   .median())
trend_stable = (df.loc[df["churned_next_30d"] == 0, "watch_trend_7d_to_30d"]
                  .median())
spread_pp = (
    df.groupby("plan_tier")["churned_next_30d"].mean().max() -
    df.groupby("plan_tier")["churned_next_30d"].mean().min()
) * 100
heatmap_gap = heatmap.loc["m2 (trial)", "casual"] - heatmap.loc["m25+", "heavy"]

print(f"\nKey patterns for Phase 3 feature engineering:")
print(f"  - Plan tier spread:        {spread_pp:.2f}pp churn-rate difference")
print(f"  - Engagement trend median: churners {trend_decline:.3f} vs "
      f"non-churners {trend_stable:.3f}  (<1 = declining)")
print(f"  - Tenure x cohort max gap: m2 casual ({heatmap.loc['m2 (trial)', 'casual']*100:.1f}%) "
      f"vs m25+ heavy ({heatmap.loc['m25+', 'heavy']*100:.1f}%) "
      f"= {heatmap_gap*100:.1f}pp spread")
print(f"\nReady for Phase 3 (feature engineering).")
