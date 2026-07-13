"""Shared fixtures for the churn-retention test suite."""
import numpy as np
import pandas as pd
import pytest

from src.data.simulate import simulate_subscribers, SimConfig
from src.features.transforms import build_features


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """A small (2k row) deterministic subscriber dataset for fast tests."""
    return simulate_subscribers(SimConfig(n_subscribers=2000, seed=42))


@pytest.fixture(scope="session")
def engineered_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Same 2k rows with engineered features applied."""
    return build_features(raw_df)


@pytest.fixture(scope="session")
def small_probas() -> np.ndarray:
    """Hand-crafted P(churn) array for policy tests."""
    return np.array([0.02, 0.10, 0.30, 0.60, 0.90])


@pytest.fixture(scope="session")
def small_ltv() -> np.ndarray:
    """Matching LTV array for the same 5 users."""
    return np.array([72.0, 72.0, 140.0, 140.0, 228.0])
