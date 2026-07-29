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
# # Phase 1 — Data Profiling & Realism Verification
#
# Validates the synthetic StreamFlix subscriber dataset **before** any modeling. If the
# data has structural issues, fix them here rather than discovering them mid-modeling.
#
# ## Sections
#
# | Section | Check |
# |---|---|
# | **A. Schema integrity** | Nulls, duplicate IDs, memory footprint |
# | **B. Distribution profiles** | Engagement, tickets, payments, revenue |
# | **C. Nested-counts math** | Verify Poisson-thinning windows (7d ≤ 30d ≤ 90d) |
# | **D. Engagement cohort** | Bimodal split + heavy/casual realism check |
# | **E. Tenure spikes** | Verify m2 (trial drop) and m12 (anniversary) elevated churn |
# | **F. Univariate signal** | Correlation-based feature screen |
# | **G. Verdict** | Data ready for Phase 2? |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
from pathlib import Path

# Run from project root whether invoked as `python notebooks/01_...` or
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
# ## A. Schema integrity — nulls, duplicate IDs, memory

# %%
print("=" * 70)
print("A. SCHEMA INTEGRITY")
print("=" * 70)
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

null_counts = df.isnull().sum()
nulls = null_counts[null_counts > 0]
if len(nulls) == 0:
    print("No nulls in any column.  PASS")
else:
    print(f"Nulls detected:\n{nulls}")

dup_ids = df["subscriber_id"].duplicated().sum()
print(f"Duplicate subscriber IDs: {dup_ids}  "
      f"{'PASS' if dup_ids == 0 else 'FAIL'}")


# %% [markdown]
# ## B. Distribution profiles
#
# Summary stats + a grid of histograms so any weirdness (long tails, spikes at zero,
# unexpected discreteness) is obvious at a glance.

# %%
print("\n" + "=" * 70)
print("B. DISTRIBUTION PROFILES")
print("=" * 70)
for col in ["tenure_months", "watch_hours_last_30d", "logins_last_30d",
            "support_tickets_90d", "payment_failures_180d",
            "days_since_last_login", "monthly_revenue"]:
    s = df[col]
    print(f"\n{col}:")
    print(f"  mean={s.mean():.2f}  std={s.std():.2f}  "
          f"min={s.min():.0f}  p50={s.median():.1f}  max={s.max():.0f}")

# Visualize the same features so any weirdness (long tails, mode at zero,
# discreteness) is obvious at a glance
profile_cols = [
    ("tenure_months",        "hist", 30, "Tenure (months)"),
    ("watch_hours_last_30d", "hist", 40, "Watch hours (last 30d)"),
    ("logins_last_30d",      "hist", 30, "Logins (last 30d)"),
    ("days_since_last_login","hist", 30, "Days since last login"),
    ("support_tickets_90d",  "bar",  None, "Support tickets (90d)"),
    ("payment_failures_180d","bar",  None, "Payment failures (180d)"),
    ("monthly_revenue",      "bar",  None, "Monthly revenue ($)"),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for ax, (col, kind, bins, title) in zip(axes, profile_cols):
    s = df[col]
    if kind == "hist":
        ax.hist(s, bins=bins, color="#5B8FF9", edgecolor="white", alpha=0.85)
        ax.axvline(s.median(), color="#F6AD55", linestyle="--",
                   linewidth=1.5, label=f"median={s.median():.1f}")
        ax.legend(fontsize=8)
    else:
        vc = s.value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values,
               color="#5B8FF9", edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(col, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
axes[-1].set_visible(False)
plt.suptitle("Feature Distributions -- Section B Profiling",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_feature_distributions.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/01_feature_distributions.png")


# %% [markdown]
# ## C. Nested-counts math verification
#
# The simulator uses Poisson thinning so short-window counts must always ≤ long-window
# counts. If any row violates this, the multi-window features are broken.

# %%
print("\n" + "=" * 70)
print("C. NESTED-WINDOWS MATH (must hold: 7d <= 30d <= 90d, etc.)")
print("=" * 70)

checks = [
    ("support_tickets",     "support_tickets_7d",    "support_tickets_30d",   "support_tickets_90d"),
    ("payment_failures",    "payment_failures_30d",  "payment_failures_90d",  "payment_failures_180d"),
]
all_passed = True
for label, c_short, c_mid, c_long in checks:
    bad_1 = (df[c_short] > df[c_mid]).sum()
    bad_2 = (df[c_mid] > df[c_long]).sum()
    status = "PASS" if (bad_1 == 0 and bad_2 == 0) else "FAIL"
    print(f"  {label}:  {c_short} <= {c_mid} <= {c_long}   "
          f"violations: {bad_1}, {bad_2}   {status}")
    if status == "FAIL":
        all_passed = False

# Watch hours -- the 7d and 30d can in principle exceed 90d when trend > 0,
# because they are not strict subsets but trend-adjusted snapshots. So we
# only sanity-check ordering on the *medians*.
print(f"  watch_hours medians: "
      f"7d={df['watch_hours_last_7d'].median():.1f}  "
      f"30d={df['watch_hours_last_30d'].median():.1f}  "
      f"90d={df['watch_hours_last_90d'].median():.1f}  "
      "(7d is per-week so smaller; 30d ~ 90d when trend=0)")


# %% [markdown]
# ## D. Engagement cohort realism
#
# Streaming products have a bimodal engagement distribution (heavy watchers + casual
# dippers, thin middle). This section verifies our synthetic data produces that shape.

# %%
print("\n" + "=" * 70)
print("D. ENGAGEMENT COHORT REALISM")
print("=" * 70)
cohort_share = df["engagement_cohort"].value_counts(normalize=True).round(3)
print(f"Cohort share:\n{cohort_share}")

heavy_real = (df["watch_hours_last_30d"] > 30).mean()
casual_real = (df["watch_hours_last_30d"] < 3).mean()
print(f"\nHeavy users (>30 hrs/mo, last 30d):  {heavy_real:.1%}  (target: 10-15%)")
print(f"Casual users (<3 hrs/mo, last 30d):  {casual_real:.1%}  (target: 15-30%)")

# Plot: overlaid histograms per cohort
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"heavy": "#5B8FF9", "regular": "#5AD8A6", "casual": "#F6BD16"}
for cohort, color in colors.items():
    vals = df.loc[df["engagement_cohort"] == cohort, "watch_hours_last_30d"]
    ax.hist(vals.clip(upper=60), bins=40, alpha=0.55,
            label=f"{cohort} (n={len(vals):,})",
            color=color, edgecolor="white")
ax.set_xlabel("watch_hours_last_30d (clipped at 60)")
ax.set_ylabel("Subscribers")
ax.set_title("Engagement distribution by cohort (bimodal target)",
             fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_engagement_by_cohort.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/01_engagement_by_cohort.png")


# %% [markdown]
# ## E. Tenure-spike verification
#
# Verify the two churn spikes the PM brief calls out: **month 2** (post-trial drop) and
# **month 12** (annual reassessment).

# %%
print("\n" + "=" * 70)
print("E. TENURE SPIKE VERIFICATION (m2 trial drop, m12 anniversary)")
print("=" * 70)
tenure_curve = (
    df.groupby("tenure_months")["churned_next_30d"]
      .agg(["mean", "count"])
      .reset_index()
)
# Print the months around the expected spikes
print("\nChurn rate by tenure month (selected):")
for m in [0, 1, 2, 3, 4, 6, 10, 11, 12, 13, 18, 24, 36]:
    row = tenure_curve[tenure_curve["tenure_months"] == m]
    if not row.empty:
        rate = row["mean"].iloc[0]
        n = row["count"].iloc[0]
        flag = " <-- expected spike" if m in (2, 11, 12) else ""
        print(f"  m={m:>2}  rate={rate:.2%}  n={n:>5,}{flag}")

# Plot
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(tenure_curve["tenure_months"], tenure_curve["mean"] * 100,
        color="#5B8FF9", linewidth=2)
ax.axvline(2, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m2 trial drop (expected spike)")
ax.axvline(12, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m12 anniversary (expected spike)")
ax.set_xlabel("Tenure months")
ax.set_ylabel("30-day churn rate (%)")
ax.set_title("Churn rate by tenure -- verify embedded spikes",
             fontweight="bold")
ax.set_xlim(0, 36)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_tenure_churn_curve.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/01_tenure_churn_curve.png")


# %% [markdown]
# ## F. Univariate churn signal screen
#
# Correlation with `churned_next_30d` for every candidate feature. Not a model — a
# first-pass screen to see which features are worth engineering further in Phase 3.

# %%
print("\n" + "=" * 70)
print("F. UNIVARIATE CHURN SIGNAL SCREEN")
print("=" * 70)
print("(>= 0.10 = strong, 0.05-0.10 = moderate, < 0.05 = weak)\n")

numeric_cols = [
    "tenure_months", "monthly_revenue",
    "watch_hours_last_7d", "watch_hours_last_30d", "watch_hours_last_90d",
    "distinct_titles_7d", "distinct_titles_30d", "distinct_titles_90d",
    "days_since_last_login", "logins_last_30d",
    "support_tickets_7d", "support_tickets_30d", "support_tickets_90d",
    "payment_failures_30d", "payment_failures_90d", "payment_failures_180d",
    "days_since_plan_change", "days_until_promo_expires",
]
bool_cols = ["auto_renew", "multi_profile", "promo_active"]

corrs = []
for col in numeric_cols:
    corrs.append((col, df[col].corr(df["churned_next_30d"])))
for col in bool_cols:
    corrs.append((col, df[col].astype(int).corr(df["churned_next_30d"])))

corrs.sort(key=lambda x: abs(x[1]), reverse=True)
print("(>= 0.10 = STRONG, 0.05-0.10 = moderate, < 0.05 = weak)\n")
for col, corr in corrs:
    flag = "STRONG" if abs(corr) > 0.10 else (
           "moderate" if abs(corr) > 0.05 else "weak")
    print(f"  {col:<26}  corr = {corr:+.3f}  {flag}")

# Visualize the same correlations as a sorted bar chart
corr_df = pd.DataFrame(corrs, columns=["feature", "corr"])
corr_df["abs_corr"] = corr_df["corr"].abs()
corr_df = corr_df.sort_values("abs_corr", ascending=True)

fig, ax = plt.subplots(figsize=(10, 9))
colors_bar = ["#5B8FF9" if c < 0 else "#F6735B" for c in corr_df["corr"]]
ax.barh(corr_df["feature"], corr_df["corr"], color=colors_bar,
        alpha=0.85, edgecolor="white")
ax.axvspan(-1, -0.10, alpha=0.07, color="red")
ax.axvspan(0.10, 1, alpha=0.07, color="red")
for x in [-0.10, 0.10]:
    ax.axvline(x, color="gray", linestyle="--", linewidth=1)
for x in [-0.05, 0.05]:
    ax.axvline(x, color="lightgray", linestyle=":", linewidth=1)
ax.axvline(0, color="black", linewidth=0.6)
for y, (feat, c) in enumerate(zip(corr_df["feature"], corr_df["corr"])):
    offset = 0.005 if c >= 0 else -0.005
    ha = "left" if c >= 0 else "right"
    ax.text(c + offset, y, f"{c:+.3f}", va="center", ha=ha, fontsize=8)
ax.set_xlabel("Pearson correlation with churn (next 30d)")
ax.set_title("Univariate signal screen -- sorted by strength\n"
             "blue=stabilizer  |  orange=driver  |  red shading=STRONG band",
             fontweight="bold", fontsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_xlim(-0.16, 0.16)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_univariate_correlations.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/01_univariate_correlations.png")


# %% [markdown]
# ## G. Verdict — is the data ready for Phase 2?
#
# Roll-up of the 7 checks above. All must pass before proceeding to EDA + modeling.

# %%
print("\n" + "=" * 70)
print("G. VERDICT -- DATA READY FOR PHASE 2?")
print("=" * 70)

dup_ids = df["subscriber_id"].duplicated().sum()
nulls_total = df.isnull().sum().sum()
heavy_real = (df["watch_hours_last_30d"] > 30).mean()
casual_real = (df["watch_hours_last_30d"] < 3).mean()
churn_rate = df["churned_next_30d"].mean()
m2_rate = tenure_curve.loc[tenure_curve["tenure_months"] == 2, "mean"].iloc[0]
m6_rate = tenure_curve.loc[tenure_curve["tenure_months"] == 6, "mean"].iloc[0]
nested_ok = (
    (df["support_tickets_7d"] <= df["support_tickets_30d"]).all() and
    (df["support_tickets_30d"] <= df["support_tickets_90d"]).all() and
    (df["payment_failures_30d"] <= df["payment_failures_90d"]).all() and
    (df["payment_failures_90d"] <= df["payment_failures_180d"]).all()
)
checks = [
    ("No nulls", nulls_total == 0),
    ("No duplicate IDs", dup_ids == 0),
    ("Nested counts hold", nested_ok),
    ("Heavy share 10-15%", 0.10 <= heavy_real <= 0.15),
    ("Casual share 15-30%", 0.15 <= casual_real <= 0.30),
    ("Churn rate 4-8%", 0.04 <= churn_rate <= 0.08),
    ("m2 spike visible", m2_rate > m6_rate),
]
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")

if all(ok for _, ok in checks):
    print("\nAll 7 checks pass. Data is workplace-realistic and ready for Phase 2.")
else:
    print("\nOne or more checks failed. Investigate before proceeding.")
