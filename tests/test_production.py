"""Consistency tests for src/models/production.py -- the single source
of truth for what "the production model" is.

If any of these tests fail, the Streamlit app / notebooks / MLflow
Registry are at risk of getting out of sync.
"""
from pathlib import Path

from src.models.production import (
    CHURN_MODEL_PATH,
    CHURN_MODEL_REGISTRY_NAME,
    CHURN_MODEL_ARTIFACT_KEY,
    CHURN_MODEL_MLFLOW_URI,
    UPLIFT_MODEL_PATH,
    UPLIFT_MODEL_REGISTRY_NAME,
    UPLIFT_MODEL_ARTIFACT_KEY,
    UPLIFT_MODEL_MLFLOW_URI,
    UPLIFT_FOCUS_LEVER,
)


def test_paths_are_pathlib_and_end_in_pkl():
    for p in (CHURN_MODEL_PATH, UPLIFT_MODEL_PATH):
        assert isinstance(p, Path), f"{p} should be a Path, got {type(p)}"
        assert p.suffix == ".pkl", f"{p} should end in .pkl"


def test_paths_live_under_models_dir():
    for p in (CHURN_MODEL_PATH, UPLIFT_MODEL_PATH):
        assert p.parts[0] == "models", (
            f"{p} should live under models/ so gitignore + Streamlit "
            f"bootstrap logic apply consistently"
        )


def test_registry_names_follow_streamflix_convention():
    for name in (CHURN_MODEL_REGISTRY_NAME, UPLIFT_MODEL_REGISTRY_NAME):
        assert name.startswith("streamflix_"), (
            f"Registry name {name!r} should be prefixed 'streamflix_' "
            f"for shared naming across models"
        )


def test_uplift_lever_name_flows_through_all_constants():
    """The focus lever appears in both the pickle path and the Registry
    name -- if we ever switch to a different lever, these must stay
    aligned."""
    assert UPLIFT_FOCUS_LEVER in UPLIFT_MODEL_REGISTRY_NAME
    assert UPLIFT_FOCUS_LEVER in str(UPLIFT_MODEL_PATH)


def test_mlflow_uris_use_at_production_alias():
    """We use the modern @production alias (MLflow 2.9+), not the
    legacy /Production stage path."""
    for uri in (CHURN_MODEL_MLFLOW_URI, UPLIFT_MODEL_MLFLOW_URI):
        assert uri.startswith("models:/"), f"{uri} should start with models:/"
        assert "@production" in uri, (
            f"{uri} should use the @production alias, not a stage path"
        )


def test_artifact_keys_are_stable_strings():
    """These are the dict keys the pickles use. Changing them would break
    every consumer of load_production_churn_model() / load_production_uplift_model()."""
    assert CHURN_MODEL_ARTIFACT_KEY == "production_model", (
        "CHURN_MODEL_ARTIFACT_KEY is what Streamlit + Phase 6/7 read; "
        "changing it would silently break them"
    )
    assert UPLIFT_MODEL_ARTIFACT_KEY == "model"


def test_loaders_raise_helpful_error_when_pickle_missing(tmp_path, monkeypatch):
    """If someone calls the loader before running the training notebook,
    they should get a message that tells them which notebook to run."""
    import src.models.production as prod

    monkeypatch.setattr(prod, "CHURN_MODEL_PATH", tmp_path / "does_not_exist.pkl")
    monkeypatch.setattr(prod, "UPLIFT_MODEL_PATH", tmp_path / "also_missing.pkl")

    import pytest
    with pytest.raises(FileNotFoundError, match="notebooks/04_modeling.py"):
        prod.load_production_churn_model()
    with pytest.raises(FileNotFoundError, match="notebooks/08_uplift_modeling.py"):
        prod.load_production_uplift_model()
