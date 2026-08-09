"""StreamFlix Retention -- interactive decision-support app.

Two views:
    1. Overview: sidebar controls for budget, uplift, cost knobs;
       shows policy summary and comparison against blanket baseline
    2. Per-user lookup: pick a subscriber_id, see their risk score,
       recommended lever, and top model drivers

The app imports directly from src/ so the math is the same as the
notebooks. If pytest is green in CI, the demo is correct.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

# Make src/ importable when running with `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data.loader import load_subscribers
from src.features.transforms import build_features
from src.models.train import prepare_features
from src.models.production import (
    CHURN_MODEL_PATH, CHURN_MODEL_ARTIFACT_KEY,
    load_production_churn_model,
)
from src.decisions.policy import (
    INTERVENTION_MENU, LTV_BY_TIER, PREMIUM_UPGRADE_CAP_PCT,
    score_all_levers, pick_best_lever, apply_budget_cap,
    apply_premium_cap, summarize_policy, simulate_blanket_campaign,
)

# --- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="StreamFlix Retention",
    page_icon="💸",
    layout="wide",
)


# --- Bootstrap: generate data + model if missing (for Streamlit Cloud) ----
def _bootstrap_if_needed() -> None:
    """First-boot setup for Streamlit Cloud, where the repo is cloned
    fresh and data/ and models/ are gitignored. Uses the same path
    constant as everywhere else via src/models/production.py, so the
    bootstrap and the main load stay in sync automatically."""
    data_path = Path("data/subscribers.csv")

    if not data_path.exists():
        with st.spinner("First-boot setup: generating synthetic dataset..."):
            from src.data.simulate import simulate_subscribers, SimConfig
            data_path.parent.mkdir(parents=True, exist_ok=True)
            simulate_subscribers(SimConfig()).to_csv(data_path, index=False)

    if not CHURN_MODEL_PATH.exists():
        with st.spinner("First-boot setup: training model (this runs once)..."):
            from sklearn.model_selection import train_test_split
            from src.models.train import (
                train_logistic_regression, train_hist_gbm,
                tune_hist_gbm_optuna, calibrate_model,
            )
            from src.models.evaluate import compute_metrics
            raw = load_subscribers(str(data_path))
            df = build_features(raw)
            X, y = prepare_features(df)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42,
            )
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42,
            )
            # Trains the SAME model Phase 4 ships to production: LR baseline
            # + calibrated HistGradientBoosting (default vs Optuna-tuned,
            # winner selected on TEST PR-AUC). Kept in sync with
            # 04_modeling.py so a fresh Streamlit Cloud boot produces
            # byte-identical artifacts.
            lr = train_logistic_regression(X_train, y_train)
            hgb_default = train_hist_gbm(X_train, y_train)
            hgb_tuned, _best_params = tune_hist_gbm_optuna(
                X_train, y_train, X_calib, y_calib,
                n_trials=25, random_state=42,
            )
            # Pick winner on TEST PR-AUC (same rule Phase 4 uses)
            default_pr = compute_metrics(
                y_test, hgb_default.predict_proba(X_test)[:, 1])["pr_auc"]
            tuned_pr = compute_metrics(
                y_test, hgb_tuned.predict_proba(X_test)[:, 1])["pr_auc"]
            hgb = hgb_tuned if tuned_pr > default_pr else hgb_default
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


# --- Load everything once ------------------------------------------------
@st.cache_resource
def load_pipeline():
    _bootstrap_if_needed()
    raw = load_subscribers("data/subscribers.csv")
    df = build_features(raw)
    X, y = prepare_features(df)
    # Load via src/models/production.py so we're guaranteed to be reading
    # the same artifact Phase 4 wrote and Phase 6/7 also read from.
    model, artifact = load_production_churn_model()
    X = X[artifact["feature_names"]]
    p_churn = model.predict_proba(X)[:, 1]
    ltv = df["plan_tier"].map(LTV_BY_TIER).values
    tenure = df["tenure_months"].values
    return df, p_churn, ltv, tenure


try:
    df, p_churn, ltv, tenure = load_pipeline()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    error_msg = str(e)


# --- Header --------------------------------------------------------------
st.title("💸 StreamFlix Retention Targeting")
st.markdown(
    "Interactive tool for the Retention team. Adjust intervention costs, "
    "uplifts, and budget to see how the recommendation changes."
)

if not data_loaded:
    st.error(
        f"Couldn't load the model or dataset. Run "
        f"`python src/data/simulate.py` and `python notebooks/04_modeling.py` "
        f"first.\n\n**Details:** {error_msg}"
    )
    st.stop()


# --- Sidebar controls ----------------------------------------------------
st.sidebar.header("Policy knobs")

st.sidebar.subheader("Budget cap")
budget = st.sidebar.slider(
    "Monthly budget ($k)", min_value=1, max_value=500,
    value=200, step=5,
) * 1000

st.sidebar.subheader("Intervention menu")
st.sidebar.caption("Cost per user + assumed uplift on P(churn).")

menu_ui = {}
for name, defaults in INTERVENTION_MENU.items():
    st.sidebar.markdown(f"**{name}**")
    c1, c2 = st.sidebar.columns(2)
    cost = c1.number_input(
        f"cost ${name}", min_value=0.0, max_value=50.0,
        value=float(defaults["cost"]), step=0.5,
        key=f"cost_{name}", label_visibility="collapsed",
    )
    uplift = c2.number_input(
        f"uplift {name}", min_value=0.0, max_value=1.0,
        value=float(defaults["uplift"]), step=0.01,
        format="%.2f",
        key=f"uplift_{name}", label_visibility="collapsed",
    )
    menu_ui[name] = {"cost": cost, "uplift": uplift}

st.sidebar.subheader("Baseline comparison")
blanket_month = st.sidebar.slider(
    "Blanket target tenure month", min_value=1, max_value=24, value=11,
)
blanket_cost = st.sidebar.number_input(
    "Blanket $ per user", min_value=0.0, max_value=50.0,
    value=5.0, step=0.5,
)


# --- Compute policies ----------------------------------------------------
policy = pick_best_lever(p_churn, ltv, menu_ui)
policy = apply_budget_cap(policy, budget=budget)
policy = apply_premium_cap(policy, n_total=len(policy),
                           cap_pct=PREMIUM_UPGRADE_CAP_PCT)
targeted_summary = summarize_policy(policy, targeted_only=True)

baseline_summary = simulate_blanket_campaign(
    p_churn=p_churn, ltv=ltv, tenure_months=tenure,
    cost_per_user=blanket_cost, target_month=blanket_month,
    uplift=menu_ui["credit_5"]["uplift"],
)


# --- Layout: two tabs ----------------------------------------------------
tab_overview, tab_lookup = st.tabs(["📊 Policy overview", "🔍 Per-user lookup"])

with tab_overview:
    # KPI row
    st.markdown("### Impact vs current-state blanket campaign")
    c1, c2, c3, c4 = st.columns(4)
    delta_ev = (targeted_summary["net_expected_value"]
                - baseline_summary["net_expected_value"])
    c1.metric(
        "Users targeted",
        f"{targeted_summary['n_targeted']:,}",
        f"{targeted_summary['n_targeted'] - baseline_summary['n_targeted']:,}",
    )
    c2.metric(
        "Total cost",
        f"${targeted_summary['total_cost']:,.0f}",
        f"${targeted_summary['total_cost'] - baseline_summary['total_cost']:,.0f}",
        delta_color="inverse",
    )
    c3.metric(
        "Net EV / month",
        f"${targeted_summary['net_expected_value']:,.0f}",
        f"${delta_ev:,.0f}",
    )
    c4.metric(
        "ROI",
        f"{targeted_summary['roi_multiplier']:.2f}x",
        f"{targeted_summary['roi_multiplier'] - baseline_summary['roi_multiplier']:+.2f}x",
    )

    # Side-by-side comparison table
    st.markdown("### Head-to-head")
    def _fmt(row, v):
        if row == "roi_multiplier":
            return f"{v:.2f}x"
        if row == "n_targeted":
            return f"{v:,.0f}"
        return f"${v:,.0f}"

    compare = pd.DataFrame({
        row: {c: _fmt(row, v[row])
              for c, v in [("Blanket baseline", baseline_summary),
                           ("Targeted policy", targeted_summary)]}
        for row in ["n_targeted", "total_cost",
                    "expected_retained_revenue",
                    "net_expected_value", "roi_multiplier"]
    }).T
    compare.index = ["Users contacted", "Total cost",
                     "Expected retained revenue",
                     "Net expected value", "ROI multiplier"]
    st.dataframe(compare, use_container_width=True)

    # Lever mix
    if targeted_summary["n_targeted"] > 0:
        st.markdown("### Lever mix (targeted only)")
        lever_mix = (policy[policy["will_target"]]["best_lever"]
                     .value_counts().rename_axis("lever")
                     .reset_index(name="users"))
        lever_mix["share"] = (lever_mix["users"] /
                              lever_mix["users"].sum()).map("{:.1%}".format)
        st.dataframe(lever_mix, use_container_width=True, hide_index=True)


with tab_lookup:
    st.markdown("### Look up a specific subscriber")

    df_lookup = df.copy()
    df_lookup["p_churn"] = p_churn
    df_lookup["best_lever"] = policy["best_lever"].values
    df_lookup["best_ev"] = policy["best_ev"].values
    df_lookup["will_target"] = policy["will_target"].values

    # Simple selector: subscriber_id
    sub_id = st.selectbox(
        "Subscriber ID",
        options=df_lookup["subscriber_id"].sort_values().tolist(),
        index=int(np.argmax(p_churn)),
    )
    row = df_lookup[df_lookup["subscriber_id"] == sub_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(churn) 30d", f"{row['p_churn']:.1%}")
    c2.metric("Plan", row["plan_tier"])
    c3.metric("Tenure (months)", row["tenure_months"])
    c4.metric("Recommended", row["best_lever"])

    st.markdown("### Why the model flagged this user")
    diag_cols = [
        "watch_hours_last_7d", "watch_hours_last_30d",
        "days_since_last_login", "logins_last_30d",
        "support_tickets_7d", "support_tickets_30d",
        "payment_failures_30d", "payment_failures_90d",
        "days_since_plan_change", "days_until_promo_expires",
        "auto_renew", "engagement_cohort",
    ]
    diag = pd.DataFrame({
        "feature": diag_cols,
        "value": [row[c] for c in diag_cols],
    })
    st.dataframe(diag, use_container_width=True, hide_index=True)

    if row["will_target"]:
        st.success(
            f"**Action:** apply `{row['best_lever']}` "
            f"— cost ${menu_ui[row['best_lever']]['cost']:.2f}, "
            f"expected value ${row['best_ev']:.2f}"
        )
    else:
        st.info(
            "**Action:** no intervention. "
            "No lever has positive expected value for this user."
        )


st.markdown("---")
st.caption(
    "Model: HistGradientBoosting (Optuna-tuned when tuning holds up on test) "
    "+ Platt calibration. "
    "Full methodology in `notebooks/06_decision_rule.ipynb`. "
    "Decision memo in `reports/decision_memo.md`."
)
