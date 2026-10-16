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
# # Phase 9 — Head-to-Head: blanket vs v1 (propensity) vs v2 (uplift)
#
# Phase 6 ships the **v1 policy**: `P(churn) × assumed uplift × LTV − cost`. Phase 8
# trained a **v2 uplift model** that predicts each user's response to `credit_5`
# directly. This notebook runs both, plus today's blanket campaign, an oracle and a
# random policy, and scores them against the simulator's ground truth: `true_uplift`,
# the churn reduction the credit would really cause for each user.
#
# **Fair comparison rules.**
# - **One lever** (`credit_5`) for every policy, since v2 only exists for that lever.
# - **Same spend.** Policies are compared at equal budgets, so "more value" can't
#   come from simply contacting more people.
# - **Held-out users only.** Evaluation uses the subscribers that neither the churn
#   model (Phase 4 test split) nor the uplift model (Phase 8 test split) trained on.
# - **Ground truth, not outcomes.** `true_uplift × LTV` is the expected retained
#   revenue per user, so the comparison has no outcome noise; it only exists because
#   the data is simulated.
#
# ## Sections
#
# | Section | Purpose |
# |---|---|
# | **A. Setup** | Score held-out users with both models |
# | **B. Ranking quality** | How well does each score rank true uplift? |
# | **C. Value at equal spend** | True net value across budget levels |
# | **D. Each policy's own targeting rule** | What happens when each targets every user it thinks is positive-EV |
# | **E. Verdict** | Recommendation |

# %%
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

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
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from src.data.loader import load_subscribers_experiment
from src.features.transforms import build_features
from src.models.train import prepare_features
from src.models.production import (
    load_production_churn_model, load_production_uplift_model, UPLIFT_FOCUS_LEVER,
)
from src.decisions.policy import INTERVENTION_MENU, LTV_BY_TIER, sample_budget

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

LEVER = UPLIFT_FOCUS_LEVER
COST = INTERVENTION_MENU[LEVER]["cost"]
ASSUMED_UPLIFT = INTERVENTION_MENU[LEVER]["uplift"]


# %% [markdown]
# ## A. Setup — held-out users, both models

# %%
print("=" * 70)
print(f"A. SETUP (lever: {LEVER}, cost ${COST:.0f})")
print("=" * 70)

raw = load_subscribers_experiment("data/subscribers_experiment.csv")
df = build_features(raw)
X, y = prepare_features(df)
t = df["treated"].values

# Recreate both training splits to find rows neither model saw
idx = np.arange(len(df))
_, churn_test = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)
_, uplift_test = train_test_split(idx, test_size=0.30, stratify=t, random_state=42)
held_out = np.intersect1d(churn_test, uplift_test)
ev_df = df.iloc[held_out].reset_index(drop=True)
X_ho = X.iloc[held_out]

churn_model, churn_art = load_production_churn_model()
uplift_model, uplift_art = load_production_uplift_model()
p_churn = churn_model.predict_proba(X_ho[churn_art["feature_names"]])[:, 1]
# sklift predicts P(churn|treated) - P(churn|control); flip so positive = churn reduced
pred_lift = -uplift_model.predict(X_ho[uplift_art["feature_names"]])

ltv = ev_df["plan_tier"].map(LTV_BY_TIER).values
true_value = ev_df["true_uplift"].values * ltv  # expected retained revenue if treated
tenure = ev_df["tenure_months"].values
n = len(ev_df)

print(f"Held-out users (in neither training set): {n:,}")
print(f"Mean true uplift: {ev_df['true_uplift'].mean():+.4f}  "
      f"(sleeping dogs: {(ev_df['true_uplift'] < 0).mean():.1%})")
print(f"True value of a credit, averaged over users: ${true_value.mean():.2f} "
      f"vs cost ${COST:.0f}")
print(f"Users for whom the credit truly pays (true_uplift x LTV > ${COST:.0f}): "
      f"{(true_value > COST).mean():.1%}")

# Each policy's targeting score (its own estimate of value per user)
scores = {
    "v1 propensity (P(churn) x 15%)": p_churn * ASSUMED_UPLIFT * ltv,
    "v2 uplift model": pred_lift * ltv,
    "Oracle (true uplift)": true_value,
}
rng = np.random.default_rng(0)
scores["Random"] = rng.random(n)


# %% [markdown]
# ## B. Ranking quality
#
# A policy is only as good as its ranking: which users does it put first? Rank
# correlation with the true value per user, and the true value captured in each
# score's top decile.

# %%
print("\n" + "=" * 70)
print("B. RANKING QUALITY")
print("=" * 70)
print(f"\n  {'score':<32} {'Spearman vs truth':>18} {'top-10% mean true value':>24}")
top = int(0.10 * n)
for name, s in scores.items():
    rho = spearmanr(s, true_value).statistic
    top_val = true_value[np.argsort(-s)[:top]].mean()
    print(f"  {name:<32} {rho:>18.2f} {'$' + f'{top_val:.2f}':>24}")
print(f"\n  (Average over all users: ${true_value.mean():.2f} per user.)")


# %% [markdown]
# ## C. True net value at equal spend
#
# For each budget, each policy credits its top-ranked users until the money runs out
# (a positive estimated value is not required, so every policy spends the same amount).
# The blanket campaign is shown at its own fixed spend. Budgets are expressed on the
# full-base scale (2.1M subscribers) and scaled to this held-out sample.

# %%
print("\n" + "=" * 70)
print("C. TRUE NET VALUE AT EQUAL SPEND")
print("=" * 70)


def value_at_budget(score: np.ndarray, budget: float) -> float:
    k_users = int(budget // COST)
    chosen = np.argsort(-score)[:k_users]
    return float(true_value[chosen].sum() - COST * len(chosen))


scale = 2_100_000 / n
pop_budgets = np.array([25, 50, 100, 200, 400, 800]) * 1000.0
rows = []
for b in pop_budgets:
    b_s = sample_budget(n, b)
    rows.append({"budget": b, **{name: value_at_budget(s, b_s) * scale
                                  for name, s in scores.items()}})
curve = pd.DataFrame(rows)
short = {"v1 propensity (P(churn) x 15%)": "v1", "v2 uplift model": "v2",
         "Oracle (true uplift)": "oracle", "Random": "random"}
print("\nTrue net value per month on the full base ($k), credit_5 only:")
print(f"  {'budget':>8}" + "".join(f"{short[c]:>10}" for c in scores))
for _, r in curve.iterrows():
    print(f"  {r['budget'] / 1000:>7.0f}k" + "".join(
        f"{r[c] / 1000:>10.1f}" for c in scores))

blanket_mask = tenure == 11
blanket_value = (true_value[blanket_mask].sum() - COST * blanket_mask.sum()) * scale
blanket_spend = COST * blanket_mask.sum() * scale
v1_at_blanket = value_at_budget(scores["v1 propensity (P(churn) x 15%)"],
                                COST * blanket_mask.sum()) * scale
print(f"\nBlanket m11: spend ${blanket_spend / 1000:,.0f}k -> true net value "
      f"${blanket_value / 1000:,.1f}k. v1 with the same spend: ${v1_at_blanket / 1000:,.1f}k.")

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = {"v1 propensity (P(churn) x 15%)": "#5B8FF9", "v2 uplift model": "#F6735B",
          "Oracle (true uplift)": "#5AD8A6", "Random": "#A0A0A0"}
for name in scores:
    ax.plot(curve["budget"] / 1000, curve[name] / 1000, marker="o", linewidth=2,
            color=colors[name], label=name)
ax.scatter([blanket_spend / 1000], [blanket_value / 1000], color="#F6BD16", s=90,
           zorder=3, label="Blanket m11 (fixed spend)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xscale("log")
ax.set_xlabel("Monthly credit_5 budget, full base ($k, log scale)")
ax.set_ylabel("True net value ($k / month)")
ax.set_title("credit_5 targeting at equal spend, scored against ground truth",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "09_v1_vs_v2_headtohead.png", dpi=140, bbox_inches="tight")
print(f"\nSaved -> {FIG_DIR}/09_v1_vs_v2_headtohead.png")


# %% [markdown]
# ## D. Each policy's own targeting rule (no budget)
#
# Without a budget, each policy targets every user it believes is positive-EV. Overly
# optimistic scores lead to over-targeting, which shows up as low or negative true value.

# %%
print("\n" + "=" * 70)
print("D. EACH POLICY'S OWN TARGETING RULE")
print("=" * 70)
print(f"\n  {'policy':<32} {'users (full base)':>17} {'spend':>10} {'true net value':>15}")
for name in ["v1 propensity (P(churn) x 15%)", "v2 uplift model", "Oracle (true uplift)"]:
    chosen = scores[name] > COST
    spend = COST * chosen.sum() * scale
    val = (true_value[chosen].sum() - COST * chosen.sum()) * scale
    print(f"  {name:<32} {chosen.sum() * scale:>17,.0f} {'$' + f'{spend / 1000:,.0f}k':>10} "
          f"{'$' + f'{val / 1000:,.1f}k':>15}")


# %% [markdown]
# ## E. Verdict

# %%
print("\n" + "=" * 70)
print("E. VERDICT")
print("=" * 70)
at_200 = curve[curve["budget"] == 200_000].iloc[0]
v1n, v2n = "v1 propensity (P(churn) x 15%)", "v2 uplift model"
print(f"""
At a $200k/month credit_5 budget (full base): v1 ${at_200[v1n] / 1000:,.1f}k, v2 ${at_200[v2n] / 1000:,.1f}k,
random ${at_200['Random'] / 1000:,.1f}k, oracle ${at_200['Oracle (true uplift)'] / 1000:,.1f}k of true net value per month.

- v1 ranks users far better than v2. In this simulation the credit's effect is
  proportional to a user's churn risk, which the churn model estimates well from
  50k labelled users; the uplift model has to learn the same thing from a 0.7pp
  average effect in 25k-per-arm data, and can't.
- The experiment is still valuable: it measures the average effect of credit_5
  (Phase 8, Section A), which is the input v1 needs. Use it to replace the 15%
  assumption, and re-run uplift modelling only when a larger or repeated experiment
  gives it enough signal.
- Recommendation: keep v1. Do not ship v2.
""")
