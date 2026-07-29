"""Unit tests for Phase 8 uplift modeling: trainers, evaluation
metrics, and the uplift-aware decision policy."""
import numpy as np
import pandas as pd
import pytest

from src.models.uplift import (
    train_s_learner, train_t_learner, train_x_learner,
    train_class_transformation, predict_uplift,
)
from src.models.evaluate import compute_uplift_metrics, qini_curve_points
from src.decisions.policy import pick_best_lever_uplift, INTERVENTION_MENU


# ---------------------------------------------------------------------
# Small synthetic fixture: 1500 users, 6 features, known heterogeneity
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def uplift_data():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
        "f4": rng.uniform(0, 1, n),
        "f5": rng.integers(0, 5, n).astype(float),
        "f6": rng.integers(0, 2, n).astype(float),
    })
    t = (rng.random(n) < 0.5).astype(int)

    # Baseline churn probability depends on f1, f2
    logit = -1.0 + 0.8 * X["f1"] + 0.5 * X["f2"]
    p_control = 1.0 / (1.0 + np.exp(-logit))

    # Treatment effect is HETEROGENEOUS by f3: high-f3 users benefit MORE
    # (bigger churn reduction). Low-f3 users are neutral / slightly worse.
    treatment_effect = 0.15 * X["f3"]  # can be negative for low f3

    p_treated = np.clip(p_control - treatment_effect, 0.001, 0.999)
    y_control = rng.binomial(1, p_control)
    y_treated = rng.binomial(1, p_treated)
    y = np.where(t == 1, y_treated, y_control)

    return X, y, t.astype(int)


# ---------------------------------------------------------------------
# Trainers: shape + fit-doesn't-crash
# ---------------------------------------------------------------------
@pytest.mark.parametrize("trainer_fn", [
    train_s_learner, train_t_learner,
    train_x_learner, train_class_transformation,
])
def test_uplift_trainer_returns_per_user_predictions(uplift_data, trainer_fn):
    X, y, t = uplift_data
    model = trainer_fn(X, t, y)
    u = predict_uplift(model, X)
    assert u.shape == (len(X),), (
        f"expected {len(X)} predictions, got shape {u.shape}"
    )
    assert np.isfinite(u).all(), "predictions contain non-finite values"


def test_t_learner_catches_heterogeneity(uplift_data):
    """T-learner should assign LOWER predicted churn uplift (=more retention
    lift) to users with high f3, since our fixture wired heterogeneity
    into that feature. Weak but non-trivial signal expected."""
    X, y, t = uplift_data
    model = train_t_learner(X, t, y)
    u = predict_uplift(model, X)

    # In sklift convention u = P(Y=1|T=1) - P(Y=1|T=0). Lower u for
    # high-f3 users means their churn drops more under treatment.
    high_f3 = X["f3"] > X["f3"].quantile(0.75)
    low_f3 = X["f3"] < X["f3"].quantile(0.25)
    assert u[high_f3].mean() < u[low_f3].mean(), (
        "T-learner failed to catch f3 heterogeneity: "
        f"high f3 mean uplift={u[high_f3].mean():.4f}, "
        f"low f3 mean uplift={u[low_f3].mean():.4f}"
    )


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------
def test_qini_auc_is_finite_and_signed(uplift_data):
    X, y, t = uplift_data
    model = train_t_learner(X, t, y)
    u = predict_uplift(model, X)
    m = compute_uplift_metrics(y, u, t)
    assert np.isfinite(m["qini_auc"])
    assert np.isfinite(m["retention_lift_at_30pct"])
    assert np.isfinite(m["retention_lift_at_10pct"])
    # A trained model should beat random ranking on the data it was
    # fit on (in-sample; this is a weak sanity check, not generalization)
    assert m["qini_auc"] > 0.0, (
        f"trained T-learner scored qini_auc={m['qini_auc']:.4f} on "
        "training data -- expected > 0"
    )


def test_qini_curve_points_shape(uplift_data):
    X, y, t = uplift_data
    model = train_t_learner(X, t, y)
    u = predict_uplift(model, X)
    qc = qini_curve_points(y, u, t)
    assert set(qc.columns) == {"share_targeted", "cumulative_retention_lift"}
    assert (qc["share_targeted"] >= 0).all()
    assert (qc["share_targeted"] <= 1.0 + 1e-9).all()
    # Monotonic in share_targeted
    assert (qc["share_targeted"].diff().dropna() >= 0).all()


def test_random_uplift_scores_near_zero_qini():
    """A random uplift score should produce Qini AUC near 0."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.binomial(1, 0.1, size=n)
    t = rng.binomial(1, 0.5, size=n)
    u_random = rng.normal(0, 1, size=n)
    m = compute_uplift_metrics(y, u_random, t)
    assert abs(m["qini_auc"]) < 0.05, (
        f"random ranker Qini AUC = {m['qini_auc']:.4f}, expected near 0"
    )


# ---------------------------------------------------------------------
# Uplift-aware policy
# ---------------------------------------------------------------------
def test_pick_best_lever_uplift_shape_and_dtypes():
    n = 100
    ltv = np.full(n, 100.0)
    uplift_by_lever = {
        "curated_playlist": np.full(n, 0.05),
        "credit_5":         np.full(n, 0.10),
    }
    policy = pick_best_lever_uplift(uplift_by_lever, ltv)
    assert len(policy) == n
    assert set(policy.columns) == {"best_lever", "best_ev", "cost"}
    # credit_5 has higher uplift * ltv - cost, should win everywhere
    assert (policy["best_lever"] == "credit_5").all()


def test_pick_best_lever_uplift_selects_none_for_sleeping_dogs():
    """Users with negative retention lift on every lever should get 'none'."""
    n = 50
    ltv = np.full(n, 100.0)
    # First 25 users are sleeping dogs (negative uplift on all levers)
    uplift_by_lever = {
        "curated_playlist": np.concatenate([np.full(25, -0.1), np.full(25, 0.05)]),
        "credit_5":         np.concatenate([np.full(25, -0.1), np.full(25, 0.10)]),
    }
    policy = pick_best_lever_uplift(uplift_by_lever, ltv)
    # Sleeping dogs → 'none'
    assert (policy.iloc[:25]["best_lever"] == "none").all(), (
        "sleeping dogs (negative uplift) should get 'none' from uplift policy"
    )
    assert (policy.iloc[:25]["cost"] == 0.0).all()
    # Persuadables → an actual lever
    assert (policy.iloc[25:]["best_lever"] != "none").all()


def test_pick_best_lever_uplift_beats_pick_best_lever_on_sleeping_dogs():
    """Regression: the whole point of the uplift policy is that it skips
    sleeping dogs. The propensity-only policy would target a high-P(churn)
    sleeping dog; the uplift policy should not."""
    from src.decisions.policy import pick_best_lever

    n = 20
    ltv = np.full(n, 200.0)
    # A user with high churn probability but NEGATIVE uplift = classic
    # sleeping dog. Propensity-only sees high P(churn) and targets them;
    # uplift-aware sees negative lift and skips them.
    p_churn = np.full(n, 0.5)  # high across the board
    uplift_by_lever = {
        "credit_5": np.concatenate([
            np.full(10, -0.05),  # sleeping dogs: treatment hurts
            np.full(10, +0.15),  # persuadables: treatment helps
        ]),
    }

    prop_policy = pick_best_lever(p_churn, ltv, INTERVENTION_MENU)
    up_policy = pick_best_lever_uplift(uplift_by_lever, ltv, INTERVENTION_MENU)

    n_sleep_prop = int((prop_policy.iloc[:10]["best_lever"] != "none").sum())
    n_sleep_up = int((up_policy.iloc[:10]["best_lever"] != "none").sum())
    assert n_sleep_up < n_sleep_prop, (
        f"uplift policy should target FEWER sleeping dogs than propensity "
        f"policy (uplift targets {n_sleep_up}/10, propensity targets "
        f"{n_sleep_prop}/10)"
    )
