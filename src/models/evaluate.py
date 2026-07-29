"""Model evaluation utilities.

Two layers:
    1. compute_metrics(y_true, y_proba) -- the standard tier of scores
       (PR-AUC, ROC-AUC, Brier, log-loss). Returns a flat dict so it can
       be dropped straight into a comparison DataFrame.
    2. top_k_metrics(y_true, y_proba, k) -- decision-rule-relevant
       precision/recall at top-K probability cutoffs (proxy for
       "if I target my top 10%, what fraction will churn?").

Why both PR-AUC AND ROC-AUC?
    PR-AUC focuses on the positive (churn) class, which is the right
    primary metric for an imbalanced (~5% positive) decision-cost
    problem. ROC-AUC is the lingua franca and lets us compare to
    published benchmarks.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    brier_score_loss, log_loss,
    precision_score, recall_score,
)


def compute_metrics(y_true, y_proba) -> Dict[str, float]:
    """Compute the standard tier of binary-classification scores."""
    return {
        "pr_auc":  average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "brier":   brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def top_k_metrics(y_true, y_proba, k: float = 0.10) -> Dict[str, float]:
    """Precision and recall when targeting the top-K fraction by probability.

    Mirrors the decision-rule mechanics in Phase 6: 'if the Retention
    team can only afford to contact the top K% of users, how many real
    churners would they reach (recall) and what fraction of their
    contacts would be real churners (precision)?'
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    k_count = int(np.ceil(n * k))
    # Indices of the top-K highest predicted probabilities
    top_idx = np.argsort(y_proba)[-k_count:]
    y_pred = np.zeros(n, dtype=int)
    y_pred[top_idx] = 1
    return {
        "k": k,
        "k_count": k_count,
        "precision_at_k": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_k": recall_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------
# Uplift-model evaluation (Phase 8)
# ---------------------------------------------------------------------
def compute_uplift_metrics(y_true, uplift, treatment,
                           strategy: str = "overall") -> Dict[str, float]:
    """Compute the standard tier of uplift-model scores.

    We evaluate on the RETENTION framing (positive uplift = retained
    users saved) rather than the churn framing (negative uplift = churn
    reduced). This keeps every metric interpretable as "higher is better."

    Returns:
        qini_auc              -- Qini AUC on retention (analog of ROC-AUC
                                 for uplift). Higher = better; 0 = random.
        retention_lift_at_30  -- Absolute retention lift when targeting
                                 the top 30% of users by predicted lift.
                                 Reads as "additional retained users per
                                 targeted user, vs. targeting at random."
        retention_lift_at_10  -- Same at top-10% (tighter budget).

    Sign convention:
        We take uplift as the model's predicted P(churn|T=1)-P(churn|T=0)
        (sklift-native, negative for retention treatments). Internally we
        flip both the outcome (y_retained = 1 - y_churn) and the score
        (predicted_retention_lift = -predicted_churn_uplift) so sklift's
        "higher is better" metrics apply directly.
    """
    try:
        from sklift.metrics import qini_auc_score, uplift_at_k
    except ImportError:
        raise ImportError(
            "scikit-uplift is required for uplift evaluation. "
            "Install with `pip install scikit-uplift`."
        )
    y_churn = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)

    y_retained = 1 - y_churn
    retention_score = -uplift

    qini = qini_auc_score(y_retained, retention_score, treatment)
    u30 = uplift_at_k(y_retained, retention_score, treatment,
                      strategy=strategy, k=0.30)
    u10 = uplift_at_k(y_retained, retention_score, treatment,
                      strategy=strategy, k=0.10)
    return {
        "qini_auc": float(qini),
        "retention_lift_at_30pct": float(u30),
        "retention_lift_at_10pct": float(u10),
    }


def qini_curve_points(y_true, uplift, treatment) -> pd.DataFrame:
    """Return the (share_targeted, cumulative_retention_lift) points
    that make up a Qini curve. Feed the columns straight into matplotlib.

    Uses the retention framing (see compute_uplift_metrics): higher
    curves = better model. A diagonal line = random targeting; a curve
    above the diagonal = the model beats random.
    """
    try:
        from sklift.metrics import qini_curve
    except ImportError:
        raise ImportError(
            "scikit-uplift is required. `pip install scikit-uplift`."
        )
    y_churn = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)

    y_retained = 1 - y_churn
    x, y = qini_curve(y_retained, -uplift, treatment)
    n = len(y_churn)
    return pd.DataFrame({
        "share_targeted": x / max(n, 1),
        "cumulative_retention_lift": y,
    })


def calibration_curve_points(y_true, y_proba, n_bins: int = 10):
    """Bin probabilities into n_bins quantile bins and compute
    (mean predicted prob, fraction positive) per bin.

    Implemented manually so we can use QUANTILE bins (equal-population)
    instead of sklearn's default equal-width bins, which leave most
    bins empty when probabilities cluster low (typical for imbalanced
    problems).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    quantile_edges = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0 + 1e-9  # ensure max value is included

    bin_idx = np.digitize(y_proba, quantile_edges[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_pred": float(y_proba[mask].mean()),
            "frac_positive": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)
