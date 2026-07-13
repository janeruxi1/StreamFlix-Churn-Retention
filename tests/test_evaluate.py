"""Model evaluation metric tests."""
import numpy as np
import pytest

from src.models.evaluate import (
    compute_metrics, top_k_metrics, calibration_curve_points,
)


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.2, 0.6])
    m = compute_metrics(y_true, y_proba)
    assert set(m.keys()) == {"pr_auc", "roc_auc", "brier", "log_loss"}


def test_compute_metrics_ranges():
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.2, 0.6])
    m = compute_metrics(y_true, y_proba)
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["brier"] <= 1
    assert m["log_loss"] >= 0


def test_top_k_metrics_matches_definition():
    """Users are ranked descending by proba; top-K flagged as positive."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.05])
    m = top_k_metrics(y_true, y_proba, k=0.40)
    # k=0.40 of 10 = 4 users flagged
    assert m["k_count"] == 4
    # top 4 by proba are indices 1, 3, 5, 7 -> all real churners
    assert m["precision_at_k"] == 1.0
    # 4 real churners total, we caught all 4 -> recall = 1.0
    assert m["recall_at_k"] == 1.0


def test_top_k_metrics_at_full_coverage():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = top_k_metrics(y_true, y_proba, k=1.0)
    # k=1.0 -> flag everyone -> precision = base rate, recall = 1
    assert m["recall_at_k"] == 1.0
    assert m["precision_at_k"] == 0.5


def test_calibration_curve_points_returns_dataframe_with_bins():
    y_true = np.random.default_rng(42).binomial(1, 0.1, size=1000)
    y_proba = np.random.default_rng(42).uniform(0, 1, size=1000)
    curve = calibration_curve_points(y_true, y_proba, n_bins=5)
    assert len(curve) == 5
    assert set(curve.columns) == {"bin", "n", "mean_pred", "frac_positive"}


def test_calibration_curve_perfect_model_hugs_diagonal():
    """If predictions == truth, each bin's mean_pred == frac_positive."""
    rng = np.random.default_rng(42)
    y_proba = rng.uniform(0, 1, size=10000)
    y_true = (rng.uniform(0, 1, size=10000) < y_proba).astype(int)
    curve = calibration_curve_points(y_true, y_proba, n_bins=10)
    # For a perfect model, mean_pred and frac_positive should be within 0.05
    diffs = np.abs(curve["mean_pred"] - curve["frac_positive"])
    assert diffs.max() < 0.05, f"max calibration gap = {diffs.max():.3f}"
