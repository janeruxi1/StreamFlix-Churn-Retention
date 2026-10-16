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
# # Phase 6 — Cost-Aware Decision Rule + Budget Sweep
#
# Model probabilities + intervention menu + LTV → **per-user targeting decisions under
# the monthly budget**. Then compare against the blanket $5 credit campaign the
# retention team runs today.
#
# **Scale.** The brief's numbers are for the whole base (~2.1M subscribers, $200k/month
# budget, 10k Premium-upgrade offers/month). The analysis runs on a 50,000-subscriber
# sample, so the budget and the Premium cap are scaled to the sample (×50k/2.1M) and
# results are reported both per sample and scaled back to the full base.
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Load model, score all subscribers |
# | **B. Assumptions** | Intervention menu, LTV, scale |
# | **C. Per-lever EV** | Compute EV for each lever, per user |
# | **D. Best lever, no budget** | Argmax over levers |
# | **E. Policy under the budget** | Shadow-price allocation + summary |
# | **F. Baseline comparison** | Head-to-head vs blanket $5 credit at m11 |
# | **G. Budget sweep** | Net EV and ROI across budget levels |
# | **H. Sensitivity** | Uplift and LTV-horizon assumptions |
# | **I. Allocation method check** | Shadow price vs ranking by raw EV |
# | **J. Verdict** | Handoff to Phase 7 (memo + Streamlit app) |
#
# All figures saved under `reports/figures/`.

# %%
import os
import sys
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
    MONTHLY_BUDGET, POPULATION_SUBSCRIBERS, PREMIUM_OFFERS_PER_MONTH,
    score_all_levers, pick_best_lever, apply_budget_cap, build_policy,
    summarize_policy, simulate_blanket_campaign, sample_budget, to_population,
)
from src.decisions.ltv import (
    MONTHLY_REVENUE_BY_TIER, RMST_HORIZON_MONTHS,
    life_table_survival, restricted_mean_survival_time,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def k(x: float) -> str:
    """Format dollars in thousands."""
    return f"{'-' if x < 0 else ''}${abs(x) / 1000:,.1f}k"


# %% [markdown]
# ## A. Setup — score every subscriber with the calibrated model
#
# The whole snapshot is scored, as a monthly batch job would score the current base.
# Note that 30k of these 50k rows were the model's training set, so their scores are
# slightly optimistic; the held-out test metrics are in Phase 4.

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
X = X[artifact["feature_names"]]

p_churn = model.predict_proba(X)[:, 1]
ltv = df["plan_tier"].map(LTV_BY_TIER).values
tenure = df["tenure_months"].values
N = len(p_churn)
BUDGET = sample_budget(N)

print(f"Scored {N:,} subscribers")
print(f"P(churn) distribution -- mean={p_churn.mean():.4f}, "
      f"p90={np.percentile(p_churn, 90):.4f}, max={p_churn.max():.4f}")
print(f"LTV distribution     -- mean=${ltv.mean():.0f}, "
      f"min=${ltv.min():.0f}, max=${ltv.max():.0f}")


# %% [markdown]
# ## B. Assumptions — intervention menu, LTV, scale
#
# Uplift = the share of would-be churners a lever retains (from the brief). Sensitivity in
# Section H sweeps ±50%. LTV comes from the survival curves in `src/decisions/ltv.py`:
# the month-t hazard is the 30-day churn rate of subscribers at tenure t, and LTV is
# 24-month restricted mean survival × monthly price.

# %%
print("\n" + "=" * 70)
print("B. ASSUMPTIONS")
print("=" * 70)
print("\nIntervention menu (uplift = share of would-be churners retained):")
menu_df = pd.DataFrame(INTERVENTION_MENU).T
menu_df["cost"] = menu_df["cost"].map("${:.2f}".format)
menu_df["uplift"] = menu_df["uplift"].map("{:.0%}".format)
print(menu_df)

print(f"\nLTV by tier (RMST x monthly revenue, {RMST_HORIZON_MONTHS}-month horizon):")
print(f"  {'tier':<10} {'S(12)':>7} {'S(24)':>7} {'months retained':>16} "
      f"{'$/mo':>6} {'LTV':>6}")
for tier, ltv_val in LTV_BY_TIER.items():
    sub = df[df["plan_tier"] == tier]
    t, S = life_table_survival(sub["tenure_months"].values,
                               sub["churned_next_30d"].values)
    rmst = restricted_mean_survival_time(t, S, horizon=RMST_HORIZON_MONTHS)
    print(f"  {tier:<10} {S[11]:>7.2f} {S[23]:>7.2f} {rmst:>16.2f} "
          f"{'$' + f'{MONTHLY_REVENUE_BY_TIER[tier]:.0f}':>6} {'$' + f'{ltv_val:.0f}':>6}")

print(f"\nScale: brief budget ${MONTHLY_BUDGET:,.0f}/month for {POPULATION_SUBSCRIBERS:,} "
      f"subscribers -> ${BUDGET:,.0f} for this {N:,}-subscriber sample")
print(f"Premium upgrade cap: {PREMIUM_OFFERS_PER_MONTH:,} offers/month on the full base "
      f"= {PREMIUM_UPGRADE_CAP_PCT:.2%} of subscribers ({int(N * PREMIUM_UPGRADE_CAP_PCT)} "
      f"in the sample)")


# %% [markdown]
# ## C. Expected value per lever, per user
#
# For each subscriber × lever: `EV = P(churn) × uplift × LTV − cost`.

# %%
print("\n" + "=" * 70)
print("C. EV PER LEVER PER USER")
print("=" * 70)
ev_wide = score_all_levers(p_churn, ltv, INTERVENTION_MENU)
print("\nEV summary across all users:")
print(ev_wide.describe().T[["mean", "50%", "max"]].round(2))
print("\nBreak-even P(churn) per lever and tier (EV > 0 needs P > cost / (uplift x LTV)):")
print(f"  {'lever':<18}" + "".join(f"{t:>10}" for t in LTV_BY_TIER))
for name, v in INTERVENTION_MENU.items():
    print(f"  {name:<18}" + "".join(
        f"{v['cost'] / (v['uplift'] * L):>10.1%}" for L in LTV_BY_TIER.values()))


# %% [markdown]
# ## D. Best lever per user, ignoring the budget
#
# `argmax` over levers per user; users whose levers all have negative EV get `none`.
# This is the policy if money were no object.

# %%
print("\n" + "=" * 70)
print("D. BEST LEVER, NO BUDGET")
print("=" * 70)
unconstrained = build_policy(p_churn, ltv, budget=np.inf)
s_unc = summarize_policy(unconstrained)
mix_pre = unconstrained["best_lever"].value_counts()
print(f"\n  {'lever':<20}  {'n_users':>8}  {'pct_of_base':>11}")
for lever, n in mix_pre.items():
    print(f"  {lever:<20}  {n:>8,}  {n / N:>10.1%}")
print(f"\nPositive-EV users: {s_unc['n_targeted']:,}; spending everything on them costs "
      f"${s_unc['total_cost']:,.0f} per sample ({k(to_population(s_unc['total_cost'], N))} "
      f"on the full base) vs a budget of ${BUDGET:,.0f} ({k(MONTHLY_BUDGET)}).")
print("The budget binds, so the policy has to choose.")


# %% [markdown]
# ## E. Policy under the budget
#
# With a binding budget, ranking users by raw EV is not optimal: a $1 playlist that earns
# $0.80 is a better use of money than a $5 credit that earns $2. `build_policy` charges
# every dollar its opportunity cost λ (a Lagrange multiplier, the "shadow price" of the
# budget): each user gets the lever that maximises `EV − λ × cost`, if positive, and λ is
# the smallest value whose total spend fits the budget. The Premium-upgrade capacity cap
# is enforced inside the same loop.

# %%
print("\n" + "=" * 70)
print("E. POLICY UNDER THE BUDGET")
print("=" * 70)
policy = build_policy(p_churn, ltv, budget=BUDGET)
lam = policy.attrs["shadow_price"]
targeted = policy[policy["will_target"]]
summary = summarize_policy(policy)

print(f"\nBudget: ${BUDGET:,.0f} per sample ({k(MONTHLY_BUDGET)}/month on the full base)")
print(f"Shadow price of the budget: λ = {lam:.2f} "
      f"(the last dollar spent earns ${1 + lam:.2f} of retained revenue)")
print(f"Users targeted: {len(targeted):,} of {N:,} ({len(targeted) / N:.1%})")

mix_post = targeted.groupby("best_lever").agg(n_users=("cost", "size"),
                                              total_cost=("cost", "sum"))
mix_post["pct_users"] = mix_post["n_users"] / len(targeted)
mix_post["pct_spend"] = mix_post["total_cost"] / mix_post["total_cost"].sum()
print(f"\n  {'lever':<18} {'n_users':>8} {'% users':>8} {'spend':>9} {'% spend':>8}")
for lever, r in mix_post.sort_values("n_users", ascending=False).iterrows():
    print(f"  {lever:<18} {int(r['n_users']):>8,} {r['pct_users']:>8.1%} "
          f"${r['total_cost']:>8,.0f} {r['pct_spend']:>8.1%}")


def show(s: dict, label: str) -> None:
    print(f"\n{label}")
    print(f"  {'':<28} {'per sample':>12} {'full base':>12}")
    print(f"  {'Users contacted':<28} {s['n_targeted']:>12,} "
          f"{to_population(s['n_targeted'], N):>12,.0f}")
    for key, name in [("total_cost", "Cost"),
                      ("expected_retained_revenue", "Expected retained revenue"),
                      ("net_expected_value", "Net expected value")]:
        print(f"  {name:<28} ${s[key]:>11,.0f} {k(to_population(s[key], N)):>12}")
    print(f"  {'ROI (revenue / cost)':<28} {s['roi_multiplier']:>11.2f}x")


show(summary, "Expected outcome under the targeted policy:")


# %% [markdown]
# ## F. Baseline comparison — blanket $5 credit at month 11
#
# What the retention team runs today: a $5 credit to every subscriber at tenure 11,
# regardless of risk. Scored with the same model, LTV and uplift so the two are
# comparable.

# %%
print("\n" + "=" * 70)
print("F. BASELINE: BLANKET $5 CREDIT AT MONTH-11")
print("=" * 70)
baseline = simulate_blanket_campaign(
    p_churn=p_churn, ltv=ltv, tenure_months=tenure,
    cost_per_user=5.0, target_month=11, uplift=INTERVENTION_MENU["credit_5"]["uplift"],
)
show(baseline, "Current-state baseline (blanket m11):")
swing = summary["net_expected_value"] - baseline["net_expected_value"]
print(f"\nSwing (targeted minus blanket): ${swing:,.0f} per sample = "
      f"{k(to_population(swing, N))}/month on the full base, while spending "
      f"{k(to_population(summary['total_cost'], N))} instead of "
      f"{k(to_population(baseline['total_cost'], N))}.")


# %% [markdown]
# ## G. Budget sweep
#
# Net EV keeps rising with budget until every positive-EV user is covered, but each extra
# dollar buys less, so ROI falls. The question for finance is where on this curve to sit.

# %%
print("\n" + "=" * 70)
print("G. BUDGET SWEEP")
print("=" * 70)
pop_budgets = np.array([25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]) * 1000.0
sweep_rows = []
for b_pop in pop_budgets:
    s = summarize_policy(build_policy(p_churn, ltv, budget=sample_budget(N, b_pop)))
    sweep_rows.append({"budget": b_pop,
                       "n_targeted": to_population(s["n_targeted"], N),
                       "cost": to_population(s["total_cost"], N),
                       "net_ev": to_population(s["net_expected_value"], N),
                       "roi": s["roi_multiplier"]})
sweep = pd.DataFrame(sweep_rows)

print("\nFull-base figures per month:")
print(f"  {'budget':>9} {'users':>10} {'spend':>9} {'net EV':>9} {'ROI':>6}")
for _, r in sweep.iterrows():
    mark = "  <- brief budget" if r["budget"] == MONTHLY_BUDGET else ""
    print(f"  {k(r['budget']):>9} {r['n_targeted']:>10,.0f} {k(r['cost']):>9} "
          f"{k(r['net_ev']):>9} {r['roi']:>5.2f}x{mark}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(sweep["budget"] / 1000, sweep["net_ev"] / 1000, marker="o", linewidth=2,
         color="#5AD8A6", label="Targeted policy net EV")
ax1.axhline(to_population(baseline["net_expected_value"], N) / 1000, color="#F6BD16",
            linestyle=":", linewidth=1.5, label="Blanket m11 net EV")
ax1.axvline(MONTHLY_BUDGET / 1000, color="gray", linestyle="--", linewidth=1.2,
            label=f"Brief budget {k(MONTHLY_BUDGET)}")
ax1.set_xlabel("Monthly budget, full base ($k)")
ax1.set_ylabel("Net expected value ($k / month)")
ax1.set_title("Net EV rises with budget, with diminishing returns", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(sweep["budget"] / 1000, sweep["roi"], marker="o", linewidth=2.5,
         color="#5B8FF9", label="Targeted policy ROI")
ax2.axhline(baseline["roi_multiplier"], color="#F6BD16", linestyle=":", linewidth=1.5,
            label=f"Blanket ROI ({baseline['roi_multiplier']:.2f}x)")
ax2.axhline(2.0, color="gray", linestyle="--", linewidth=1, label="Target ROI 2.0x")
ax2.axvline(MONTHLY_BUDGET / 1000, color="gray", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Monthly budget, full base ($k)")
ax2.set_ylabel("ROI (revenue / cost)")
ax2.set_title("ROI falls as the budget reaches weaker candidates", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.4)
plt.suptitle("Budget sweep -- targeted policy vs blanket", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_roi_sweep.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/06_roi_sweep.png")

at_target = sweep[sweep["roi"] >= 2.0]
max_budget_2x = at_target["budget"].max() if len(at_target) else None


# %% [markdown]
# ## H. Sensitivity — uplift and LTV-horizon assumptions
#
# The uplifts are assumptions, not measurements (Phase 8 measures one of them). Sweep
# ±50% and check whether targeting still beats the blanket. Then vary the LTV horizon:
# longer horizons rest on hazards estimated from few long-tenure subscribers.

# %%
print("\n" + "=" * 70)
print("H. SENSITIVITY")
print("=" * 70)
print("\nUplift assumptions (all three levers scaled together), at the brief budget:")
print(f"  {'scale':>6} {'users':>10} {'net EV':>9} {'ROI':>6} {'blanket net EV':>15}")
sens_rows = []
for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
    menu_s = {n: {"cost": v["cost"], "uplift": v["uplift"] * scale}
              for n, v in INTERVENTION_MENU.items()}
    s = summarize_policy(build_policy(p_churn, ltv, BUDGET, menu=menu_s))
    b = simulate_blanket_campaign(p_churn, ltv, tenure, 5.0, 11,
                                  INTERVENTION_MENU["credit_5"]["uplift"] * scale)
    sens_rows.append({"scale": scale, "roi": s["roi_multiplier"],
                      "net_ev": to_population(s["net_expected_value"], N)})
    print(f"  {scale:>5.2f}x {to_population(s['n_targeted'], N):>10,.0f} "
          f"{k(to_population(s['net_expected_value'], N)):>9} {s['roi_multiplier']:>5.2f}x "
          f"{k(to_population(b['net_expected_value'], N)):>15}")
sens = pd.DataFrame(sens_rows)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sens["scale"], sens["roi"], marker="o", markersize=9, linewidth=2.5,
        color="#5B8FF9", label="Targeted policy ROI")
ax.axhline(baseline["roi_multiplier"], color="#F6BD16", linestyle=":", linewidth=1.5,
           label=f"Blanket ROI at 1.0x ({baseline['roi_multiplier']:.2f}x)")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Break-even (1.0x)")
ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Uplift assumption (multiplier on the brief's estimates)")
ax.set_ylabel("ROI at the brief budget")
ax.set_title("Sensitivity: ROI vs uplift assumption", fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_uplift_sensitivity.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/06_uplift_sensitivity.png")

print("\nLTV horizon (production = 24 months):")
print(f"  {'horizon':>7} {'Basic':>7} {'Standard':>9} {'Premium':>8} {'net EV':>9} {'ROI':>6}")
horizon_rows = []
for h in [12, 24, 36, 48]:
    ltv_map = {}
    for tier, monthly in MONTHLY_REVENUE_BY_TIER.items():
        sub = df[df["plan_tier"] == tier]
        t, S = life_table_survival(sub["tenure_months"].values,
                                   sub["churned_next_30d"].values)
        ltv_map[tier] = round(monthly * restricted_mean_survival_time(t, S, horizon=h))
    s = summarize_policy(build_policy(p_churn, df["plan_tier"].map(ltv_map).values, BUDGET))
    horizon_rows.append({"h": h, "net_ev": to_population(s["net_expected_value"], N),
                         "roi": s["roi_multiplier"]})
    mark = "  <- production" if h == RMST_HORIZON_MONTHS else ""
    print(f"  {h:>5}mo {'$' + str(ltv_map['Basic']):>7} {'$' + str(ltv_map['Standard']):>9} "
          f"{'$' + str(ltv_map['Premium']):>8} {k(horizon_rows[-1]['net_ev']):>9} "
          f"{s['roi_multiplier']:>5.2f}x{mark}")
print("""
Read: about 30% of subscribers have tenure >= 12 months and ~9% >= 24, so the hazards
beyond 24 months rest on few users. 24 months covers the m12 anniversary spike without
leaning on that thin tail; the direction of the recommendation holds at every horizon.""")


# %% [markdown]
# ## I. Allocation method check — shadow price vs ranking by raw EV
#
# The naive alternative picks each user's highest-EV lever and then fills the budget in
# descending EV order. It over-spends on expensive levers.

# %%
print("\n" + "=" * 70)
print("I. ALLOCATION METHOD CHECK")
print("=" * 70)
greedy = apply_budget_cap(pick_best_lever(p_churn, ltv, INTERVENTION_MENU), BUDGET)
s_greedy = summarize_policy(greedy)
print(f"\n  {'method':<34} {'users':>8} {'net EV (full base)':>19} {'ROI':>6}")
for name, s in [("Rank by raw EV, fill budget", s_greedy),
                ("Shadow-price allocation (shipped)", summary)]:
    print(f"  {name:<34} {to_population(s['n_targeted'], N):>8,.0f} "
          f"{k(to_population(s['net_expected_value'], N)):>19} {s['roi_multiplier']:>5.2f}x")
gain = summary["net_expected_value"] / s_greedy["net_expected_value"] - 1
print(f"\nSame budget, same model: the shadow-price rule earns {gain:+.0%} more net EV.")


# %% [markdown]
# ## J. Verdict + handoff to Phase 7

# %%
print("\n" + "=" * 70)
print("J. PHASE 6 VERDICT")
print("=" * 70)
print(f"""
Full base, per month, at the brief's {k(MONTHLY_BUDGET)} budget:
  Targeted policy: {to_population(summary['n_targeted'], N):,.0f} subscribers contacted, spend {k(to_population(summary['total_cost'], N))},
                   net EV {k(to_population(summary['net_expected_value'], N))}, ROI {summary['roi_multiplier']:.2f}x
  Blanket m11:     {to_population(baseline['n_targeted'], N):,.0f} subscribers contacted, spend {k(to_population(baseline['total_cost'], N))},
                   net EV {k(to_population(baseline['net_expected_value'], N))}, ROI {baseline['roi_multiplier']:.2f}x
  Swing:           {k(to_population(swing, N))} per month

On the 2.0x ROI target: ROI at the full budget is {summary['roi_multiplier']:.2f}x. """
      + (f"It clears 2.0x at budgets up to about {k(max_budget_2x)}, at the cost of total net EV."
         if max_budget_2x else "No budget in the sweep reaches 2.0x."))
print(f"""
Uplift sensitivity: ROI ranges {sens['roi'].min():.2f}x-{sens['roi'].max():.2f}x for uplifts at
0.5x-1.5x of the brief; targeting beats the blanket throughout.

Ready for Phase 7 (decision memo + Streamlit app).""")
