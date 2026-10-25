"""StreamFlix Retention -- interactive decision-support app.

Tabs:
    1. Targeted vs blanket  -- the monthly business case and how it moves with budget
    2. Look up a subscriber -- risk score, the reasons behind it, and the action
    3. Assumptions          -- lever costs and uplifts, blanket-campaign settings

The app imports directly from src/, so it runs the same tested functions as
the notebooks. Figures are computed on the 50,000-subscriber sample and
scaled to the full 2.1M base.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Make src/ importable when running with `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data.loader import load_subscribers  # noqa: E402
from src.decisions.policy import (  # noqa: E402
    INTERVENTION_MENU, LTV_BY_TIER, POPULATION_SUBSCRIBERS,
    PREMIUM_UPGRADE_CAP_PCT, build_policy, sample_budget,
    simulate_blanket_campaign, summarize_policy, to_population,
)
from src.features.transforms import build_features  # noqa: E402
from src.models.explain import map_to_intervention  # noqa: E402
from src.models.production import (  # noqa: E402
    CHURN_MODEL_ARTIFACT_KEY, CHURN_MODEL_PATH, load_production_churn_model,
)
from src.models.train import prepare_features  # noqa: E402

REPO_URL = "https://github.com/janeruxi1/StreamFlix-Churn-Retention"
TEAL, ORANGE, GREY, RED = "#0F766E", "#DD6B20", "#A0AEC0", "#C53030"
LEVER_LABEL = {
    "curated_playlist": "🎬 Curated playlist",
    "credit_5": "💵 $5 credit",
    "premium_upgrade": "⭐ Premium upgrade",
    "none": "No contact",
}
LEVER_ORDER = ["curated_playlist", "credit_5", "premium_upgrade"]
# Axis labels in $k, e.g. $200k / -$250k (d3's "s" format repeats labels)
KAXIS = ("(datum.value < 0 ? '-' : '') + '$' + "
         "format(abs(datum.value) / 1000, ',.0f') + 'k'")

st.set_page_config(page_title="StreamFlix Retention", page_icon="💸", layout="wide")


def money(x: float) -> str:
    """Dollar string that never trips Streamlit's $...$ maths rendering."""
    sign = "-" if x < 0 else ""
    return f"{sign}\\${abs(x):,.0f}"


def cents(x: float) -> str:
    """Per-subscriber dollars with cents, escaped for markdown."""
    sign = "-" if x < 0 else ""
    return f"{sign}\\${abs(x):,.2f}"


def money_plain(x: float) -> str:
    """Dollar string for widgets that do not render markdown (metrics, tables)."""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def k(x: float) -> str:
    sign = "-" if x < 0 else "+"
    return f"{sign}\\${abs(x) / 1000:,.0f}k"


def nice(feature: str) -> str:
    return feature.replace("_", " ").replace(" 7d", " (7 days)").replace(
        " 30d", " (30 days)").replace(" 90d", " (90 days)").replace(
        " 180d", " (180 days)").capitalize()


# --- Bootstrap: generate data + model if missing (for Streamlit Cloud) ----
def _bootstrap_if_needed() -> None:
    """First-boot setup for Streamlit Cloud, where models/ is gitignored.
    Trains the same model notebook 04 ships: Optuna-tuned HistGradientBoosting
    with Platt calibration, on the same seeded splits."""
    data_path = Path("data/subscribers.csv")
    if not data_path.exists():
        with st.spinner("First-boot setup: generating the synthetic dataset..."):
            from src.data.simulate import SimConfig, simulate_subscribers
            data_path.parent.mkdir(parents=True, exist_ok=True)
            simulate_subscribers(SimConfig()).to_csv(data_path, index=False)

    if not CHURN_MODEL_PATH.exists():
        with st.spinner("First-boot setup: training the churn model (about a minute, once)..."):
            from sklearn.model_selection import train_test_split

            from src.models.evaluate import compute_metrics
            from src.models.train import (
                calibrate_model, train_logistic_regression, tune_hist_gbm_optuna,
            )
            df = build_features(load_subscribers(str(data_path)))
            X, y = prepare_features(df)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
            lr = train_logistic_regression(X_train, y_train)
            # Notebook 04 ships the tuned variant by design (decided before
            # scoring the test set), so the app does the same.
            hgb, _ = tune_hist_gbm_optuna(X_train, y_train, X_calib, y_calib,
                                          n_trials=25, random_state=42)
            hgb_cal = calibrate_model(hgb, X_calib, y_calib, method="sigmoid")
            CHURN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CHURN_MODEL_PATH, "wb") as f:
                pickle.dump({
                    CHURN_MODEL_ARTIFACT_KEY: hgb_cal,
                    "baseline_model": lr,
                    "feature_names": list(X.columns),
                    "metrics": {
                        "production": compute_metrics(
                            y_test, hgb_cal.predict_proba(X_test)[:, 1]),
                        "baseline": compute_metrics(
                            y_test, lr.predict_proba(X_test)[:, 1]),
                    },
                    "training_meta": {
                        "n_train": len(X_train), "n_calib": len(X_calib),
                        "n_test": len(X_test), "positive_rate": float(y.mean()),
                    },
                }, f)


@st.cache_resource(show_spinner="Loading the model and scoring 50,000 subscribers...")
def load_pipeline():
    _bootstrap_if_needed()
    df = build_features(load_subscribers("data/subscribers.csv"))
    X, _ = prepare_features(df)
    model, artifact = load_production_churn_model()
    X = X[artifact["feature_names"]]
    p_churn = model.predict_proba(X)[:, 1]
    ltv = df["plan_tier"].map(LTV_BY_TIER).values
    return df.reset_index(drop=True), X.reset_index(drop=True), model, p_churn, ltv


@st.cache_resource(show_spinner=False)
def shap_explainer(_model):
    """TreeExplainer on the uncalibrated tree inside the calibration wrapper
    (calibration is monotonic, so it does not change which features matter)."""
    import shap
    inner = _model.calibrated_classifiers_[0].estimator
    tree = getattr(inner, "estimator", inner)  # unwrap FrozenEstimator
    return shap.TreeExplainer(tree)


def menu_key(menu: dict) -> tuple:
    return tuple((n, menu[n]["cost"], menu[n]["uplift"]) for n in LEVER_ORDER)


@st.cache_data(show_spinner=False)
def budget_sweep(_p, _ltv, key: tuple) -> pd.DataFrame:
    menu = {n: {"cost": c, "uplift": u} for n, c, u in key}
    rows = []
    for b in range(25_000, 500_001, 25_000):
        s = summarize_policy(build_policy(_p, _ltv, budget=sample_budget(len(_p), b),
                                          menu=menu, premium_cap_pct=PREMIUM_UPGRADE_CAP_PCT))
        rows.append({"budget": b,
                     "net": to_population(s["net_expected_value"], len(_p)),
                     "roi": s["roi_multiplier"]})
    return pd.DataFrame(rows)


try:
    df, X, model, p_churn, ltv = load_pipeline()
except FileNotFoundError as e:
    st.error("Couldn't load the model or dataset. Run `python src/data/simulate.py` "
             f"and `python notebooks/04_modeling.py` first.\n\n**Details:** {e}")
    st.stop()
N_SAMPLE = len(p_churn)


# --- Header --------------------------------------------------------------
st.title("💸 StreamFlix Retention Targeting")
st.markdown(
    "StreamFlix sends a **\\$5 credit to every subscriber in their 11th month**. "
    "This tool replaces that blanket campaign with a targeted one: a churn model "
    "scores every subscriber, and each one gets the cheapest offer that pays for "
    "itself, or nothing, within a monthly budget. "
    "*Synthetic data; figures are per month for the full 2.1M-subscriber base.*"
)
with st.expander("ℹ️ How to use this demo"):
    st.markdown(
        """
1. **💰 Targeted vs blanket**: the monthly business case. Move the budget slider
   to see how many people to contact and what it earns.
2. **👤 Look up a subscriber**: pick an example (or type an ID) to see their churn
   risk, the reasons behind it, and the offer they get.
3. **⚙️ Assumptions**: change each offer's cost and how many would-be churners
   it keeps. Every tab updates.

**Net value** = revenue kept from subscribers who would otherwise have left,
minus the cost of the offers. **ROI** = revenue kept per dollar spent.
"""
    )

tab_case, tab_user, tab_assume = st.tabs(
    ["💰 Targeted vs blanket", "👤 Look up a subscriber", "⚙️ Assumptions"])

# --- Assumptions tab first: its widgets feed every other tab --------------
with tab_assume:
    st.markdown(
        "Only the \\$5 credit was tested in a randomised holdout. The playlist and "
        "Premium uplifts are **planning assumptions**, so try lower values and "
        "watch the business case on the first tab."
    )
    menu_ui = {}
    cols = st.columns(3)
    for col, name in zip(cols, LEVER_ORDER):
        d = INTERVENTION_MENU[name]
        with col.container(border=True):
            st.markdown(f"**{LEVER_LABEL[name]}**")
            cost = st.number_input("Cost per subscriber ($)", 0.0, 50.0,
                                   float(d["cost"]), 0.5, key=f"cost_{name}")
            uplift = st.slider("Would-be churners it keeps (%)", 0, 60,
                               int(round(d["uplift"] * 100)), 1, key=f"uplift_{name}")
            menu_ui[name] = {"cost": cost, "uplift": uplift / 100}
    st.markdown("**Current blanket campaign (the baseline)**")
    b1, b2 = st.columns(2)
    blanket_month = b1.slider("Sent in tenure month", 1, 24, 11, key="blanket_month")
    blanket_cost = b2.number_input("Credit per subscriber ($)", 0.0, 50.0, 5.0, 0.5,
                                   key="blanket_cost")
    st.caption(
        "The blanket campaign uses the \\$5 credit's uplift. Premium upgrades are "
        f"capped at {PREMIUM_UPGRADE_CAP_PCT:.2%} of subscribers a month "
        "(10,000 offers) by operations capacity."
    )

def compute(budget_full_dollars: float):
    pol = build_policy(p_churn, ltv, budget=sample_budget(N_SAMPLE, budget_full_dollars),
                       menu=menu_ui, premium_cap_pct=PREMIUM_UPGRADE_CAP_PCT)
    tgt = summarize_policy(pol, targeted_only=True)
    base = simulate_blanket_campaign(
        p_churn=p_churn, ltv=ltv, tenure_months=df["tenure_months"].values,
        cost_per_user=blanket_cost, target_month=blanket_month,
        uplift=menu_ui["credit_5"]["uplift"])
    for s in (tgt, base):
        for key in ("n_targeted", "total_cost", "expected_retained_revenue",
                    "net_expected_value"):
            s[key] = to_population(s[key], N_SAMPLE)
    return pol, tgt, base


# =====================================================================
# TAB 1 -- TARGETED VS BLANKET
# =====================================================================
with tab_case:
    st.slider("Monthly budget for the whole base ($k)", 5, 500, 200, 5, key="budget_k",
              help="The brief's budget is $200k a month. The policy spends up to "
                   "this, and less if fewer offers pay for themselves.")
    budget_full = st.session_state["budget_k"] * 1000
    policy, tgt, base = compute(budget_full)
    swing = tgt["net_expected_value"] - base["net_expected_value"]

    if swing > 0:
        st.success(
            f"**Targeting earns {k(tgt['net_expected_value'])} a month** against "
            f"{k(base['net_expected_value'])} for the blanket campaign: a "
            f"{k(swing)} swing while spending "
            f"{money(tgt['total_cost'])} instead of {money(base['total_cost'])}."
        )
    else:
        st.warning("Under these assumptions targeting does **not** beat the blanket "
                   "campaign. Check the uplifts on the Assumptions tab.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net value / month, targeted", money_plain(tgt["net_expected_value"]),
              border=True, help="Revenue kept minus offer cost, full base.")
    m2.metric("Net value / month, blanket", money_plain(base["net_expected_value"]),
              border=True, help=f"\\${blanket_cost:.0f} credit to everyone in month "
                                f"{blanket_month}.")
    m3.metric("ROI, targeted", f"{tgt['roi_multiplier']:.2f}×",
              f"{tgt['roi_multiplier'] - base['roi_multiplier']:+.2f}× vs blanket",
              border=True, help="Revenue kept per dollar spent.")
    m4.metric("Subscribers contacted", f"{tgt['n_targeted']:,.0f}", border=True,
              help=f"{tgt['n_targeted'] / POPULATION_SUBSCRIBERS:.1%} of the 2.1M base.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Spend and return, per month**")
        comp = pd.DataFrame([
            {"campaign": "Blanket", "what": "Offer cost", "dollars": base["total_cost"]},
            {"campaign": "Blanket", "what": "Revenue kept",
             "dollars": base["expected_retained_revenue"]},
            {"campaign": "Targeted", "what": "Offer cost", "dollars": tgt["total_cost"]},
            {"campaign": "Targeted", "what": "Revenue kept",
             "dollars": tgt["expected_retained_revenue"]},
        ])
        chart = alt.Chart(comp).mark_bar().encode(
            y=alt.Y("campaign:N", title=None, sort=["Targeted", "Blanket"]),
            yOffset=alt.YOffset("what:N", sort=["Revenue kept", "Offer cost"]),
            x=alt.X("dollars:Q", title="Dollars per month", axis=alt.Axis(labelExpr=KAXIS)),
            color=alt.Color("what:N", scale=alt.Scale(domain=["Revenue kept", "Offer cost"],
                                                      range=[TEAL, ORANGE]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["campaign", "what", alt.Tooltip("dollars:Q", format="$,.0f")])
        st.altair_chart(chart.properties(height=220), width="stretch")
        st.caption("The blanket campaign spends more than it keeps: most people in "
                   "month 11 were never going to leave.")
    with right:
        st.markdown("**Who gets which offer**")
        mix = (policy.loc[policy["will_target"], "best_lever"].value_counts()
               .reindex(LEVER_ORDER, fill_value=0).rename_axis("lever")
               .reset_index(name="n"))
        mix["subscribers"] = mix["n"].map(lambda v: to_population(v, N_SAMPLE))
        mix["offer"] = mix["lever"].map(LEVER_LABEL)
        mix["share"] = mix["n"] / max(1, mix["n"].sum())
        bars = alt.Chart(mix).mark_bar(color=TEAL).encode(
            y=alt.Y("offer:N", sort=[LEVER_LABEL[n] for n in LEVER_ORDER], title=None,
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X("subscribers:Q", title="Subscribers per month (full base)"),
            tooltip=["offer", alt.Tooltip("subscribers:Q", format=",.0f"),
                     alt.Tooltip("share:Q", format=".1%")])
        st.altair_chart(bars.properties(height=220), width="stretch")
        st.caption("Cheap playlists go to many moderate-risk subscribers; costly "
                   "offers are kept for high-risk, high-value ones.")

    st.markdown("**How much budget is worth spending?**")
    sweep = budget_sweep(p_churn, ltv, menu_key(menu_ui))
    base_line = base["net_expected_value"]
    line = alt.Chart(sweep).mark_line(color=TEAL, strokeWidth=3, point=True).encode(
        x=alt.X("budget:Q", title="Monthly budget", axis=alt.Axis(labelExpr=KAXIS)),
        y=alt.Y("net:Q", title="Net value per month", axis=alt.Axis(labelExpr=KAXIS)),
        tooltip=[alt.Tooltip("budget:Q", format="$,.0f", title="Budget"),
                 alt.Tooltip("net:Q", format="$,.0f", title="Net value"),
                 alt.Tooltip("roi:Q", format=".2f", title="ROI")])
    you = alt.Chart(pd.DataFrame({"budget": [budget_full],
                                  "net": [tgt["net_expected_value"]]})).mark_circle(
        size=200, color=ORANGE).encode(x="budget:Q", y="net:Q")
    blanket_rule = alt.Chart(pd.DataFrame({"y": [base_line]})).mark_rule(
        color=RED, strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart((blanket_rule + line + you).properties(height=280), width="stretch")
    st.caption(
        "Teal: targeted policy at each budget. Dashed red: the blanket campaign. "
        "Orange dot: your budget. Each extra dollar earns less than the one before, "
        "so ROI falls as the budget rises."
    )

    with st.expander("Head-to-head table"):
        table = pd.DataFrame({
            "Blanket campaign": [f"{base['n_targeted']:,.0f}", money_plain(base["total_cost"]),
                                 money_plain(base["expected_retained_revenue"]),
                                 money_plain(base["net_expected_value"]),
                                 f"{base['roi_multiplier']:.2f}×"],
            "Targeted policy": [f"{tgt['n_targeted']:,.0f}", money_plain(tgt["total_cost"]),
                                money_plain(tgt["expected_retained_revenue"]),
                                money_plain(tgt["net_expected_value"]),
                                f"{tgt['roi_multiplier']:.2f}×"],
        }, index=["Subscribers contacted", "Offer cost", "Revenue kept",
                  "Net value", "ROI"])
        st.dataframe(table, width="stretch")


# =====================================================================
# TAB 2 -- LOOK UP A SUBSCRIBER
# =====================================================================
with tab_user:
    look = df[["subscriber_id", "plan_tier", "tenure_months"]].copy()
    look["p_churn"] = p_churn
    look = pd.concat([look, policy.reset_index(drop=True)], axis=1)

    def pick(mask, by="p_churn"):
        sub = look[mask]
        return None if sub.empty else int(sub.sort_values(by, ascending=False)
                                          ["subscriber_id"].iloc[0])

    best_any = look.filter(like="ev_").max(axis=1)
    look["best_any"] = best_any
    examples = {
        "Highest churn risk": pick(look["p_churn"] > -1),
        "Gets a Premium upgrade": pick(look["best_lever"] == "premium_upgrade"),
        "Gets a $5 credit": pick(look["best_lever"] == "credit_5"),
        "Gets a curated playlist": pick(look["best_lever"] == "curated_playlist"),
        "Worth contacting, but over budget": pick(~look["will_target"] & (best_any > 0),
                                                   by="best_any"),
        "Low risk, left alone": pick((look["p_churn"] < 0.01) & ~look["will_target"],
                                     by="tenure_months"),
    }
    examples = {name: sid for name, sid in examples.items() if sid is not None}

    c1, c2 = st.columns([2, 1])
    choice = c1.radio("Pick an example subscriber", list(examples), key="example",
                      horizontal=True)
    typed = c2.text_input("…or type a subscriber ID", key="sub_id",
                          placeholder=f"e.g. {examples[choice]}")
    sub_id = examples[choice]
    if typed.strip():
        if typed.strip().isdigit() and int(typed) in set(look["subscriber_id"]):
            sub_id = int(typed)
        else:
            st.warning("That ID is not in the 50,000-subscriber sample; showing the "
                       "selected example instead.")
    i = int(look.index[look["subscriber_id"] == sub_id][0])
    row = look.loc[i]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chance of leaving in 30 days", f"{row['p_churn']:.1%}", border=True,
              help=f"Average across subscribers: {p_churn.mean():.1%}.")
    m2.metric("Plan", row["plan_tier"], border=True)
    m3.metric("Months subscribed", int(row["tenure_months"]), border=True)
    m4.metric("Lifetime value if kept", money_plain(ltv[i]), border=True,
              help="Expected revenue over the next 24 months for this plan.")

    if row["will_target"]:
        st.success(
            f"**Action: {LEVER_LABEL[row['best_lever']]}** — costs "
            f"{cents(row['cost'])}, expected to earn {cents(row['best_ev'])} net."
        )
    elif best_any[i] > 0:
        st.info(
            "**Action: no contact this month.** An offer would pay for itself "
            f"(best: {cents(best_any[i])} net), but others earn more per dollar and "
            "the budget runs out first. Raise the budget to reach them."
        )
    else:
        st.info("**Action: no contact.** No offer earns back its cost for this "
                "subscriber: their risk or value is too low.")

    with st.expander("Value of each offer for this subscriber"):
        ev_rows = pd.DataFrame({
            "Offer": [LEVER_LABEL[n] for n in LEVER_ORDER],
            "Cost": [money_plain(menu_ui[n]["cost"]) for n in LEVER_ORDER],
            "Expected net value": [f"{row[f'ev_{n}']:+,.2f}" for n in LEVER_ORDER],
        })
        st.dataframe(ev_rows, hide_index=True, width="stretch")
        st.caption("Expected net value = churn risk × share kept × lifetime value − cost.")

    st.markdown("**Why the model gives this risk score**")
    try:
        expl = shap_explainer(model)(X.iloc[[i]])
        contrib = pd.DataFrame({"feature": X.columns, "value": X.iloc[i].values,
                                "shap": expl.values[0]})
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
        contrib = contrib.head(8)
        contrib["reason"] = [f"{nice(f)} = {v:.3g}" if isinstance(v, (int, float, np.number))
                             else f"{nice(f)} = {v}"
                             for f, v in zip(contrib["feature"], contrib["value"])]
        contrib["effect"] = np.where(contrib["shap"] > 0, "Raises risk", "Lowers risk")
        contrib["angle"] = [map_to_intervention(f)["lever"] for f in contrib["feature"]]
        chart = alt.Chart(contrib).mark_bar().encode(
            y=alt.Y("reason:N", sort=contrib["reason"].tolist(), title=None,
                    axis=alt.Axis(labelLimit=320, labelOverlap=False)),
            x=alt.X("shap:Q", title="Push on churn risk (log-odds, SHAP)"),
            color=alt.Color("effect:N", scale=alt.Scale(
                domain=["Raises risk", "Lowers risk"], range=[RED, TEAL]),
                legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["reason", "effect", alt.Tooltip("shap:Q", format="+.2f"),
                     alt.Tooltip("angle:N", title="Related retention angle")])
        st.altair_chart(chart.properties(height=36 * len(contrib)), width="stretch")
        st.caption("Red bars push this subscriber's risk up, teal bars pull it down. "
                   "A value of -1 means 'never' (e.g. never changed plan). Hover a "
                   "bar for the retention angle it points to.")
    except ImportError:
        st.caption("Install `shap` to see the reasons behind each score.")


st.divider()
st.caption(
    "Model: HistGradientBoosting (Optuna-tuned) with Platt calibration, scored on a "
    "50,000-subscriber sample and scaled to 2.1M. Budget allocation: shadow-price "
    f"rule in `src/decisions/policy.py`. [Decision memo]({REPO_URL}/blob/main/reports/"
    f"decision_memo.md) · [Source on GitHub]({REPO_URL})"
)
