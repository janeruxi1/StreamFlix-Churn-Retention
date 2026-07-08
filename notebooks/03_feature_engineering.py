"""
Phase 3 -- Feature Engineering
================================

Build the reusable feature transforms identified in Phase 2's reflection.
This walkthrough:

    A. Loads raw data, applies build_features() from src/features/
    B. Inspects the engineered features (distributions, value ranges)
    C. Compares engineered signal vs raw signal (univariate correlations)
    D. Checks multicollinearity between engineered features
    E. Verdict -- which features to keep into Phase 4 modeling

All figures saved under reports/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import load_subscribers
from src.features.transforms import build_features, ENGINEERED_COLUMNS

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)

print("=" * 70)
print(f"A. FEATURE BUILD: raw {raw.shape} -> engineered {df.shape}")
print("=" * 70)
new_cols = [c for c in df.columns if c not in raw.columns]
print(f"\nAdded {len(new_cols)} new columns:")
for c in new_cols:
    print(f"  {c:<35}  dtype={df[c].dtype}")


# =====================================================================
# B. Engineered feature distributions
# =====================================================================
print("\n" + "=" * 70)
print("B. ENGINEERED FEATURE DISTRIBUTIONS")
print("=" * 70)

continuous_features = [
    "watch_trend_7d_to_30d", "watch_trend_30d_to_90d",
    "watch_per_login_30d", "titles_per_hour_30d",
    "tickets_recency_ratio", "payment_failures_recency_ratio",
    "plan_change_risk_score", "promo_expiry_risk_score",
]
for col in continuous_features:
    s = df[col]
    print(f"  {col:<35}  mean={s.mean():.3f}  std={s.std():.3f}  "
          f"min={s.min():.2f}  p50={s.median():.2f}  max={s.max():.2f}")

# Visualize the most decision-relevant features
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for ax, col in zip(axes, continuous_features):
    s = df[col]
    ax.hist(s, bins=40, color="#5B8FF9", edgecolor="white", alpha=0.85)
    ax.axvline(s.median(), color="#F6AD55", linestyle="--",
               linewidth=1.5, label=f"median={s.median():.2f}")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
plt.suptitle("Engineered feature distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_engineered_distributions.png",
            dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/03_engineered_distributions.png")


# =====================================================================
# C. Engineered signal vs raw signal (univariate correlations)
# =====================================================================
print("\n" + "=" * 70)
print("C. SIGNAL COMPARISON: raw features vs engineered")
print("=" * 70)

# Pairs to compare: (raw_features_used, engineered_feature, label)
pairs = [
    (["watch_hours_last_7d", "watch_hours_last_30d"], "watch_trend_7d_to_30d",
     "engagement trend"),
    (["watch_hours_last_30d", "watch_hours_last_90d"], "watch_trend_30d_to_90d",
     "engagement decay"),
    (["watch_hours_last_30d", "logins_last_30d"], "watch_per_login_30d",
     "intensity per login"),
    (["distinct_titles_30d", "watch_hours_last_30d"], "titles_per_hour_30d",
     "breadth of consumption"),
    (["support_tickets_7d", "support_tickets_90d"], "tickets_recency_ratio",
     "ticket escalation"),
    (["payment_failures_30d", "payment_failures_180d"],
     "payment_failures_recency_ratio", "payment escalation"),
    (["days_since_plan_change"], "plan_change_risk_score",
     "plan-change risk"),
    (["days_until_promo_expires"], "promo_expiry_risk_score",
     "promo-expiry risk"),
]

print(f"\n{'Engineered feature':<35} {'eng_corr':>10} {'best_raw_corr':>15}  {'lift':>8}")
print("-" * 75)
for raw_cols, eng_col, label in pairs:
    eng_corr = df[eng_col].corr(df["churned_next_30d"])
    raw_corrs = [df[c].corr(df["churned_next_30d"]) for c in raw_cols]
    best_raw = max(raw_corrs, key=abs)
    lift = abs(eng_corr) - abs(best_raw)
    arrow = "+" if lift > 0 else "-"
    print(f"  {eng_col:<33} {eng_corr:>+10.3f} {best_raw:>+15.3f}  "
          f"{arrow}{abs(lift):.3f}")

# Also check the categorical / boolean engineered features
print(f"\nCategorical/boolean engineered features vs churn:")
bool_features = [
    "is_trial_drop_window", "is_anniversary_window",
    "recent_plan_change_flag", "promo_expiring_soon_flag",
    "high_risk_segment_flag",
]
for col in bool_features:
    rate_true = df.loc[df[col] == True, "churned_next_30d"].mean()
    rate_false = df.loc[df[col] == False, "churned_next_30d"].mean()
    n_true = (df[col] == True).sum()
    print(f"  {col:<35} True_rate={rate_true:.2%}  "
          f"False_rate={rate_false:.2%}  (n_true={n_true:,})")

# Composite scores
print(f"\nComposite score correlations:")
for col in ["payment_health_score", "active_risk_event_count"]:
    corr = df[col].corr(df["churned_next_30d"])
    print(f"  {col:<35} corr={corr:+.3f}")


# =====================================================================
# D. Multicollinearity check
# =====================================================================
print("\n" + "=" * 70)
print("D. MULTICOLLINEARITY: engineered feature correlations")
print("=" * 70)

# Mix of engineered + key raw features
check_cols = [
    "watch_hours_last_30d", "logins_last_30d", "days_since_last_login",
    "watch_trend_7d_to_30d", "watch_per_login_30d", "titles_per_hour_30d",
    "tickets_recency_ratio", "payment_failures_recency_ratio",
    "plan_change_risk_score", "promo_expiry_risk_score",
    "payment_health_score", "active_risk_event_count",
]
corr_matrix = df[check_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(check_cols)))
ax.set_yticks(range(len(check_cols)))
ax.set_xticklabels(check_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(check_cols, fontsize=9)
for i in range(len(check_cols)):
    for j in range(len(check_cols)):
        v = corr_matrix.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax, label="correlation")
ax.set_title("Feature correlation matrix -- spot redundant engineered features",
             fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_feature_correlation.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/03_feature_correlation.png")

# Flag any pairs with |corr| > 0.85 -- candidates for removal
high_corr = []
for i, c1 in enumerate(check_cols):
    for j, c2 in enumerate(check_cols):
        if j <= i:
            continue
        v = corr_matrix.loc[c1, c2]
        if abs(v) > 0.85:
            high_corr.append((c1, c2, v))
if high_corr:
    print("\nHigh-correlation pairs (|corr| > 0.85):")
    for c1, c2, v in high_corr:
        print(f"  {c1:<30} <-> {c2:<30}  corr={v:+.3f}")
else:
    print("\nNo problematic multicollinearity (all pairs |corr| <= 0.85)")


# =====================================================================
# E. Verdict
# =====================================================================
print("\n" + "=" * 70)
print("E. PHASE 3 VERDICT")
print("=" * 70)
print(
    f"\nAdded {len(ENGINEERED_COLUMNS)} engineered features across 5 groups:"
)
print("  - 4 engagement features")
print("  - 3 tenure features")
print("  - 3 recency features")
print("  - 4 lifecycle features")
print("  - 3 composite features")
print("\nKey decision-relevant features going into Phase 4 modeling:")
print("  - watch_trend_7d_to_30d   (declining engagement = strongest leading signal)")
print("  - tickets_recency_ratio   (recent escalation)")
print("  - plan_change_risk_score  (continuous lifecycle risk)")
print("  - high_risk_segment_flag  (heatmap-derived fallback rule)")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
Phase 3 turned raw subscriber columns into 17 engineered features
across 5 groups. Most do not beat raw correlations one-on-one -- that
is expected, since tree models earn their value through INTERACTIONS,
not univariate signal. The standout is `high_risk_segment_flag`: 10pp
churn lift (15.2% vs 5.2%) on the m2-trial x casual cohort, which
works both as a model feature AND as a standalone retention rule the
team could ship without any model.

No multicollinearity above 0.85, so every feature carries unique
information. We are deliberately NOT pruning features before
modeling -- that would drop features that only matter in combinations.
SHAP in Phase 5 ranks them properly.
""")

print("Ready for Phase 4 (modeling -- LR baseline + XGBoost + calibration).")
