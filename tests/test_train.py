"""Model training utility tests. Kept lightweight -- no actual XGBoost training."""
import numpy as np
import pandas as pd
import pytest

from src.models.train import prepare_features, DROP_COLS, BOOLEAN_COLS


def test_prepare_features_returns_x_and_y(engineered_df):
    X, y = prepare_features(engineered_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)


def test_prepare_features_drops_id_and_target(engineered_df):
    X, _ = prepare_features(engineered_df)
    for col in DROP_COLS:
        assert col not in X.columns, f"prepare_features didn't drop {col}"


def test_prepare_features_target_is_binary(engineered_df):
    _, y = prepare_features(engineered_df)
    assert set(y.unique()) <= {0, 1}


def test_prepare_features_one_hot_encodes_categoricals(engineered_df):
    """After one-hot, we should see plan_tier_Basic, plan_tier_Standard, etc."""
    X, _ = prepare_features(engineered_df)
    assert "plan_tier" not in X.columns  # original dropped
    # At least one one-hot column present
    assert any(c.startswith("plan_tier_") for c in X.columns)


def test_prepare_features_bool_to_int(engineered_df):
    """Boolean columns should be cast to int for XGBoost compatibility."""
    X, _ = prepare_features(engineered_df)
    for col in BOOLEAN_COLS:
        if col in X.columns:
            assert X[col].dtype in (np.int64, np.int32, int), (
                f"{col} not int-typed after prepare_features"
            )


def test_prepare_features_no_nulls(engineered_df):
    X, _ = prepare_features(engineered_df)
    assert X.isnull().sum().sum() == 0


def test_prepare_features_all_numeric(engineered_df):
    """Post-prep matrix should be all numeric (int/float/bool)."""
    X, _ = prepare_features(engineered_df)
    for col in X.columns:
        assert np.issubdtype(X[col].dtype, np.number), (
            f"{col} is non-numeric ({X[col].dtype})"
        )
