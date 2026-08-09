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
# # Phase 6 — Cost-Aware Decision Rule + ROI Sweep
#
# Model probabilities + intervention menu + LTV → **per-user targeting decisions with a
# budget cap**. Then compare against the blanket $5 credit campaign the retention team
# runs today.
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Load model, score all subscribers |
# | **B. Assumptions** | Intervention menu, LTV, cost, uplift |
# | **C. Per-lever EV** | Compute EV for each lever, per user |
# | **D. Best-lever selection** | Argmax over levers |
# | **E. Budget cap** | Apply $200k cap + policy summary |
# | **F. Baseline comparison** | Head-to-head vs blanket $5 credit at m11 |
# | **G. ROI sweep** | Revenue / cost / net-EV across 25 budget levels |
# | **H. Sensitivity** | Uplift assumption robustness |
# | **I. Verdict** | Handoff to Phase 7 (memo + Streamlit app) |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
import pickle
from pathlib import Path

# Run from project root whether invoked as `python notebooks/06_...` or
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
from src.features.transforms import build_features
from src.models.train import prepare_features
from src.models.production import load_production_churn_model
from src.decisions.policy import (
    INTERVENTION_MENU, LTV_BY_TIER, PREMIUM_UPGRADE_CAP_PCT,
    score_all_levers, pick_best_lever, apply_budget_cap,
    apply_premium_cap, summarize_policy, simulate_blanket_campaign,
)
from src.decisions.ltv import (
    MONTHLY_REVENUE_BY_TIER, RMST_HORIZON_MONTHS,
    kaplan_meier, restricted_mean_survival_time,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## A. Setup — score every subscriber with the calibrated model

# %%
print("=" * 70)
print("A. SETUP")
print("=" * 70)

raw = load_subscribers("data/subscribers.csv")
df = build_features(raw)
X, y = prepare_features(df)

# Load via src/models/production.py so we're guaranteed to be reading
# the same artifact Phase 4 wrote and Streamlit also reads from.
model, artifact = load_production_churn_model()
feature_names = artifact["feature_names"]

# Align feature order to what the model was trained on
X = X[feature_names]

p_churn = model.predict_proba(X)[:, 1]
ltv = df["plan_tier"].map(LTV_BY_TIER).values
tenure = df["tenure_months"].values

print(f"Scored {len(p_churn):,} subscribers")
print(f"P(churn) distribution -- mean={p_churn.mean():.4f}, "
      f"p90={np.percentile(p_churn, 90):.4f}, "
      f"max={p_churn.max():.4f}")
print(f"LTV distribution     -- mean=${ltv.mean():.0f}, "
      f"min=${ltv.min():.0f}, max=${ltv.max():.0f}")


# %% [markdown]
# ## B. Assumptions — intervention menu, LTV, uplift, guardrails
#
# All numbers curated with the PM (`reports/scenario_brief.md`). Uplifts are from the
# PM's 2024 pilot; sensitivity in Section H sweeps ±50%.

# %%
print("\n" + "=" * 70)
print("B. ASSUMPTIONS")
print("=" * 70)
print("\nIntervention menu:")
menu_df = pd.DataFrame(INTERVENTION_MENU).T
menu_df["cost"] = menu_df["cost"].map("${:.2f}".format)
menu_df["uplift"] = menu_df["uplift"].map("{:.0%}".format)
print(menu_df)

print(f"\nLTV by tier (RMST x monthly revenue, "
      f"{RMST_HORIZON_MONTHS}-month horizon):")
print(f"  {'tier':<10} {'months retained':>17} {'$/mo':>8} {'LTV':>8}")
print(f"  {'-'*10} {'-'*17} {'-'*8} {'-'*8}")
for tier, ltv_val in LTV_BY_TIER.items():
    sub = df[df["plan_tier"] == tier]
    t, S = kaplan_meier(sub["tenure_months"].values,
                        sub["churned_next_30d"].values)
    rmst = restricted_mean_survival_time(t, S, horizon=RMST_HORIZON_MONTHS)
    monthly = MONTHLY_REVENUE_BY_TIER[tier]
    print(f"  {tier:<10} {rmst:>17.2f} {'$' + f'{monthly:.0f}':>8} "
          f"{'$' + f'{ltv_val:.0f}':>8}")
print(f"\nMax possible retention over the horizon: "
      f"{RMST_HORIZON_MONTHS:.0f} months (RMST ceiling)")
print(f"Premium upgrade cap: {PREMIUM_UPGRADE_CAP_PCT:.0%} of base")


# %% [markdown]
# ## C. Expected value per lever, per user
#
# For each of the 50K subscribers × 3 levers, compute
# `EV = P(churn) × uplift × LTV − cost`. Wide-format matrix that Section D reduces to
# best-lever-per-user.

# %%
print("\n" + "=" * 70)
print("C. EV PER LEVER PER USER")
print("=" * 70)
ev_wide = score_all_levers(p_churn, ltv, INTERVENTION_MENU)
print("\nEV summary across all users:")
print(ev_wide.describe().T[["mean", "50%", "max"]].round(2))


# %% [markdown]
# ## D. Best-lever selection
#
# `argmax` over levers per user. Users with all-negative EVs get `best_lever = "none"`
# (skip the intervention entirely). Distribution shows how many users route to each
# lever before the budget cap is applied.

# %%
print("\n" + "=" * 70)
print("D. BEST-LEVER SELECTION")
print("=" * 70)
policy = pick_best_lever(p_churn, ltv, INTERVENTION_MENU)
print("\nBest-lever distribution (before budget cap):")
print(policy["best_lever"].value_counts().to_string())


# %% [markdown]
# ## E. Budget cap ($200k) + policy summary
#
# Sort actionable users by EV descending, cumulative-cost cutoff at the budget line, and
# also apply the **5% premium-upgrade cap** (guardrail: can't offer too many free-tier
# upgrades without margin compression).

# %%
print("\n" + "=" * 70)
print("E. BUDGET CAP + SUMMARY")
print("=" * 70)
BUDGET = 200_000.0
policy = apply_budget_cap(policy, budget=BUDGET)
policy = apply_premium_cap(policy, n_total=len(policy),
                           cap_pct=PREMIUM_UPGRADE_CAP_PCT)

targeted = policy[policy["will_target"]]
print(f"\nBudget cap: ${BUDGET:,.0f}")
print(f"Users targeted: {len(targeted):,} of {len(policy):,} "
      f"({len(targeted)/len(policy):.1%})")
print(f"\nLever mix (targeted only):")
print(targeted["best_lever"].value_counts().to_string())

summary = summarize_policy(policy, targeted_only=True)
print(f"\nExpected outcome under the targeted policy:")
print(f"  Total cost:               ${summary['total_cost']:>12,.0f}")
print(f"  Expected retained revenue: ${summary['expected_retained_revenue']:>12,.0f}")
print(f"  Net expected value:        ${summary['net_expected_value']:>12,.0f}")
print(f"  ROI multiplier:           {summary['roi_multiplier']:>12.2f}x")


# %% [markdown]
# ## F. Baseline comparison — blanket $5 credit at month 11
#
# This is what the retention team runs today: contact every user at tenure = 11 months
# with a $5 credit, regardless of predicted churn. Simulate it under the same LTV +
# uplift assumptions so the two policies are directly comparable.

# %%
print("\n" + "=" * 70)
print("F. BASELINE: BLANKET $5 CREDIT AT MONTH-11")
print("=" * 70)
baseline = simulate_blanket_campaign(
    p_churn=p_churn, ltv=ltv, tenure_months=tenure,
    cost_per_user=5.0, target_month=11, uplift=0.15,
)
print(f"\nCurrent-state baseline:")
print(f"  Users targeted (all m11):  {baseline['n_targeted']:,}")
print(f"  Total cost:               ${baseline['total_cost']:>12,.0f}")
print(f"  Expected retained revenue: ${baseline['expected_retained_revenue']:>12,.0f}")
print(f"  Net expected value:        ${baseline['net_expected_value']:>12,.0f}")
print(f"  ROI multiplier:           {baseline['roi_multiplier']:>12.2f}x")

# Head-to-head
print("\n" + "-" * 70)
print("HEAD TO HEAD")
print("-" * 70)
def _fmt(row_name, value):
    if row_name == "roi_multiplier":
        return f"{value:.2f}x"
    if row_name == "n_targeted":
        return f"{value:,.0f}"
    return f"${value:,.0f}"

head_to_head = pd.DataFrame({
    row: {col: _fmt(row, v[row])
          for col, v in [("blanket_m11", baseline), ("targeted_policy", summary)]}
    for row in ["n_targeted", "total_cost", "expected_retained_revenue",
                "net_expected_value", "roi_multiplier"]
}).T
print(head_to_head)


# %% [markdown]
# ## G. ROI sweep across budget levels
#
# 25 budget levels from $10k to $500k. Two things worth watching: (1) net EV plateaus
# fast once we've picked the ~3% of high-EV users, (2) ROI **decays** as budget grows
# because we start targeting weaker candidates.

# %%
print("\n" + "=" * 70)
print("G. ROI SWEEP")
print("=" * 70)
budgets = np.linspace(10_000, 500_000, 25)
sweep_rows = []
for b in budgets:
    p_sweep = pick_best_lever(p_churn, ltv, INTERVENTION_MENU)
    p_sweep = apply_budget_cap(p_sweep, budget=b)
    p_sweep = apply_premium_cap(p_sweep, n_total=len(p_sweep),
                                cap_pct=PREMIUM_UPGRADE_CAP_PCT)
    s = summarize_policy(p_sweep, targeted_only=True)
    sweep_rows.append({
        "budget": b,
        "n_targeted": s["n_targeted"],
        "cost": s["total_cost"],
        "revenue": s["expected_retained_revenue"],
        "net_ev": s["net_expected_value"],
        "roi": s["roi_multiplier"],
    })
sweep = pd.DataFrame(sweep_rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(sweep["budget"] / 1000, sweep["revenue"] / 1000,
         marker="o", linewidth=2, color="#5B8FF9",
         label="Expected retained revenue")
ax1.plot(sweep["budget"] / 1000, sweep["cost"] / 1000,
         marker="s", linewidth=2, color="#F6735B",
         label="Cost spent")
ax1.plot(sweep["budget"] / 1000, sweep["net_ev"] / 1000,
         marker="^", linewidth=2, color="#5AD8A6", label="Net EV")
ax1.axvline(BUDGET / 1000, color="gray", linestyle="--",
            linewidth=1.5, label=f"chosen ${BUDGET/1000:.0f}k")
ax1.axhline(baseline["net_expected_value"] / 1000, color="#F6BD16",
            linestyle=":", linewidth=1.5,
            label=f"blanket baseline net EV")
ax1.set_xlabel("Budget cap ($k)")
ax1.set_ylabel("$ thousands")
ax1.set_title("Revenue vs cost vs net EV across budget levels",
              fontweight="bold")
ax1.legend(loc="center right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(sweep["budget"] / 1000, sweep["roi"],
         marker="o", linewidth=2.5, color="#5B8FF9",
         label="Targeted policy ROI")
ax2.axhline(baseline["roi_multiplier"], color="#F6BD16",
            linestyle=":", linewidth=1.5,
            label=f"blanket baseline ROI ({baseline['roi_multiplier']:.2f}x)")
ax2.axhline(2.0, color="gray", linestyle="--", linewidth=1,
            label="target ROI >= 2.0x")
ax2.axvline(BUDGET / 1000, color="gray", linestyle="--",
            linewidth=1.5, label=f"chosen ${BUDGET/1000:.0f}k")
ax2.set_xlabel("Budget cap ($k)")
ax2.set_ylabel("ROI multiplier")
ax2.set_title("ROI decays as budget grows (targeting weaker users)",
              fontweight="bold")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Cost-aware decision rule -- ROI sweep",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_roi_sweep.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/06_roi_sweep.png")
print("\nROI sweep table:")
sweep_display = sweep.copy()
sweep_display["budget"] = sweep_display["budget"].map("${:,.0f}".format)
sweep_display["cost"] = sweep_display["cost"].map("${:,.0f}".format)
sweep_display["revenue"] = sweep_display["revenue"].map("${:,.0f}".format)
sweep_display["net_ev"] = sweep_display["net_ev"].map("${:,.0f}".format)
sweep_display["roi"] = sweep_display["roi"].map("{:.2f}x".format)
print(sweep_display.to_string(index=False))


# %% [markdown]
# ## H. Sensitivity — uplift assumption robustness
#
# The uplift numbers are PM assumptions, not measured. Sweep ±50% and check whether the
# recommendation direction survives. If it does at 0.5×, we're not blocked on running
# the A/B experiment first. (Phase 8 *does* run that experiment and replaces the
# assumption with a learned per-user uplift.)

# %%
print("\n" + "=" * 70)
print("H. SENSITIVITY: UPLIFT ASSUMPTIONS")
print("=" * 70)
uplift_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
sens_rows = []
for scale in uplift_scales:
    menu_scaled = {
        n: {"cost": v["cost"], "uplift": v["uplift"] * scale}
        for n, v in INTERVENTION_MENU.items()
    }
    p = pick_best_lever(p_churn, ltv, menu_scaled)
    p = apply_budget_cap(p, budget=BUDGET)
    s = summarize_policy(p, targeted_only=True)
    sens_rows.append({
        "uplift_scale": f"{scale:.2f}x",
        "n_targeted": s["n_targeted"],
        "net_ev": s["net_expected_value"],
        "roi": s["roi_multiplier"],
    })
sens = pd.DataFrame(sens_rows)
print(sens.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(uplift_scales, [r["roi"] for r in sens_rows],
        marker="o", markersize=10, linewidth=2.5, color="#5B8FF9",
        label="Targeted policy ROI")
ax.axhline(baseline["roi_multiplier"], color="#F6BD16", linestyle=":",
           linewidth=1.5, label=f"blanket baseline ({baseline['roi_multiplier']:.2f}x)")
ax.axhline(2.0, color="gray", linestyle="--", linewidth=1,
           label="target ROI >= 2.0x")
ax.axvline(1.0, color="gray", linestyle=":", linewidth=1,
           label="baseline uplift assumption")
ax.set_xlabel("Uplift assumption (multiplier on PM estimate)")
ax.set_ylabel("ROI multiplier at $200k budget")
ax.set_title("Sensitivity: how much does ROI depend on uplift assumptions?",
             fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_uplift_sensitivity.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/06_uplift_sensitivity.png")


# %% [markdown]
# ## H.5. Sensitivity — LTV horizon robustness
#
# LTV is `RMST(horizon) × monthly_revenue`. Production uses a **24-month** horizon,
# but the KM curve extends further. Why not 36? 48? 60? Two reasons the shorter
# horizon is safer:
#
# 1. **KM tail is noisy.** Only ~30% of subscribers are observed to 24 months,
#    ~15% to 36, ~5% to 48, ~1% to 60. Confidence intervals on S(t) widen
#    quickly past 24 months.
# 2. **Right-censoring bias.** Subscribers we observe at long tenures are the
#    self-selected survivors of earlier churn cohorts — systematically more
#    retention-prone than an average new signup. Extending the horizon
#    over-credits those survivors' retention to the LTV of a fresh subscriber.
#
# But "shorter is safer" is only a real defense if we've LOOKED at the alternative.
# This section shows LTV and the resulting policy KPIs at 12/24/36/48/60-month
# horizons so the choice is visible, not just asserted. **The production
# recommendation still uses 24** -- lower horizons under-value retention; higher
# horizons inflate LTV based on shrinking, biased tail samples.

# %%
print("\n" + "=" * 70)
print("H.5. SENSITIVITY: LTV HORIZON ROBUSTNESS")
print("=" * 70)

horizons = [12, 24, 36, 48, 60]

# --- Per-tier RMST + LTV at each horizon --------------------------------
horizon_ltv = {}   # {horizon: {tier: LTV}}
horizon_rmst = {}  # {horizon: {tier: RMST}}
for h in horizons:
    horizon_ltv[h] = {}
    horizon_rmst[h] = {}
    for tier, monthly in MONTHLY_REVENUE_BY_TIER.items():
        sub = df[df["plan_tier"] == tier]
        t, S = kaplan_meier(sub["tenure_months"].values,
                            sub["churned_next_30d"].values)
        rmst = restricted_mean_survival_time(t, S, horizon=h)
        horizon_rmst[h][tier] = rmst
        horizon_ltv[h][tier] = round(monthly * rmst)

print("\nRMST (months retained) by tier and horizon:")
print(f"  {'horizon':>7} {'Basic':>10} {'Standard':>10} {'Premium':>10}")
print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
for h in horizons:
    r = horizon_rmst[h]
    marker = "  <- production" if h == RMST_HORIZON_MONTHS else ""
    print(f"  {h:>6}mo {r['Basic']:>10.2f} {r['Standard']:>10.2f} "
          f"{r['Premium']:>10.2f}{marker}")

print("\nImplied LTV ($) by tier and horizon:")
print(f"  {'horizon':>7} {'Basic':>10} {'Standard':>10} {'Premium':>10}")
print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
for h in horizons:
    l = horizon_ltv[h]
    marker = "  <- production" if h == RMST_HORIZON_MONTHS else ""
    print(f"  {h:>6}mo ${l['Basic']:>9.0f} ${l['Standard']:>9.0f} "
          f"${l['Premium']:>9.0f}{marker}")

# --- Policy KPIs at each horizon ----------------------------------------
horizon_rows = []
for h in horizons:
    ltv_map = horizon_ltv[h]
    ltv_h = df["plan_tier"].map(ltv_map).values
    p = pick_best_lever(p_churn, ltv_h, INTERVENTION_MENU)
    p = apply_budget_cap(p, budget=BUDGET)
    p = apply_premium_cap(p, n_total=len(p), cap_pct=PREMIUM_UPGRADE_CAP_PCT)
    s = summarize_policy(p, targeted_only=True)
    horizon_rows.append({
        "horizon_mo":  h,
        "n_targeted":  s["n_targeted"],
        "total_cost":  s["total_cost"],
        "net_ev":      s["net_expected_value"],
        "roi":         s["roi_multiplier"],
    })
horizon_df = pd.DataFrame(horizon_rows)

print("\nPolicy KPIs at $200k budget for each horizon:")
disp = horizon_df.copy()
disp["horizon"]    = disp["horizon_mo"].map(lambda x: f"{x}mo")
disp["n_targeted"] = disp["n_targeted"].map("{:,}".format)
disp["total_cost"] = disp["total_cost"].map("${:,.0f}".format)
disp["net_ev"]     = disp["net_ev"].map("${:,.0f}".format)
disp["roi"]        = disp["roi"].map("{:.2f}x".format)
print(disp[["horizon", "n_targeted", "total_cost", "net_ev", "roi"]]
      .to_string(index=False))

# --- Chart: net EV + n_targeted across horizons -------------------------
fig, ax1 = plt.subplots(figsize=(10, 5.5))
color_ev, color_n = "#5B8FF9", "#F6735B"

ax1.plot(horizon_df["horizon_mo"], horizon_df["net_ev"],
         marker="o", markersize=10, linewidth=2.5, color=color_ev,
         label="Net EV / month ($, left axis)")
ax1.axvline(RMST_HORIZON_MONTHS, color="gray", linestyle=":", linewidth=1.5,
            label=f"production horizon ({RMST_HORIZON_MONTHS}mo)")
ax1.set_xlabel("LTV horizon (months)")
ax1.set_ylabel("Net EV / month  ($)", color=color_ev)
ax1.tick_params(axis="y", labelcolor=color_ev)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.set_xticks(horizons)

ax2 = ax1.twinx()
ax2.plot(horizon_df["horizon_mo"], horizon_df["n_targeted"],
         marker="s", markersize=9, linewidth=2, color=color_n,
         label="Users targeted (right axis)", alpha=0.85)
ax2.set_ylabel("Users targeted", color=color_n)
ax2.tick_params(axis="y", labelcolor=color_n)

fig.suptitle("LTV horizon sensitivity -- how much does the decision depend "
             "on where we draw the horizon?", fontweight="bold")
# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_ltv_horizon_sensitivity.png",
            dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/06_ltv_horizon_sensitivity.png")
plt.show()

print(f"""
Read on the horizon choice (production = {RMST_HORIZON_MONTHS} months):
  * Doubling the horizon (24 -> 48) roughly doubles Net EV on paper, but
    that gain sits on top of KM tail estimates from ~5% of the population
    -- wide CI, biased toward survivors. Unreliable.
  * Halving the horizon (24 -> 12) understates LTV badly (m12 is right at
    the anniversary churn spike, not past it), collapses target counts,
    and leaves clear positive-EV users un-targeted.
  * 24 months is the point where more data doesn't buy us much precision
    but does buy us survivor bias.
""")


# %% [markdown]
# ## I. Verdict + handoff to Phase 7

# %%
print("\n" + "=" * 70)
print("I. PHASE 6 VERDICT")
print("=" * 70)
lift_over_baseline = (summary["net_expected_value"] /
                      max(baseline["net_expected_value"], 1))
print(f"\nAt $200k budget:")
print(f"  Targeted net EV:  ${summary['net_expected_value']:>12,.0f}")
print(f"  Blanket net EV:   ${baseline['net_expected_value']:>12,.0f}")
print(f"  Lift over baseline: {lift_over_baseline:.1f}x")
print(f"\nTargeted ROI: {summary['roi_multiplier']:.2f}x  "
      f"(target >= 2.0x)")
print(f"Baseline ROI: {baseline['roi_multiplier']:.2f}x")
print(f"\nReady for Phase 7 (decision memo + Streamlit app).")


# %% [markdown]
# ## Design choices worth noting

# %%
print("\n" + "=" * 70)
print("DESIGN CHOICES WORTH NOTING")
print("=" * 70)
print("""
1. Policy is a separate module from the model. src/models/ produces
   P(churn); src/decisions/policy.py decides what to do with it.
   Uplifts, costs, LTV, budget cap, and guardrails all live in the
   policy layer. Retuning the campaign doesn't require retraining.
   Standard MLOps separation: ML work stable, business logic swappable.

2. Uplift is a PM assumption, not a measurement. Numbers came from a
   2024 pilot in the scenario brief. Section H sensitivity shows the
   recommendation direction survives across +/- 50%, so we're not
   blocked -- but the v1.1 followup is A/B testing each lever in
   production.
""")


# %% [markdown]
# ## Findings worth flagging (for the Phase 7 memo)

# %%
print("\n" + "=" * 70)
print("FINDINGS WORTH FLAGGING")
print("=" * 70)
print("""
1. The current blanket campaign is losing money. $7.9k spent to save
   $1.6k = $6.3k monthly loss.

2. Budget doesn't bind. Only ~3% of subscribers have positive-EV
   interventions. Targeted spend = $4.5k vs $200k cap.

3. ROI = 1.72x, below the 2.0x primary target. Still a $9.6k monthly
   improvement over the blanket baseline. To push past 2.0x we need
   stronger model discrimination or higher-uplift levers.

4. Recommendation robust to uplift assumptions. Even at 50% weaker
   uplift, ROI stays positive and beats the blanket.

5. credit_5 dominates the lever mix. Playlist and Premium upgrades
   barely fire. Worth A/B testing to validate lever choice before
   committing to production.
""")
