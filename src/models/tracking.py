"""MLflow tracking wrappers for the StreamFlix churn pipeline.

Model training in Phase 4 fits three variants (LR, XGBoost, calibrated
XGBoost). Without tracking we'd rely on comparing screenshots of the
notebook cells to figure out which run produced the model we shipped.
MLflow tracks each run's parameters, metrics, and the fitted model
itself, so we can:

    - reproduce any earlier run from its logged params
    - compare metric deltas across runs in the MLflow UI
    - re-load the specific model artifact the Streamlit app uses

Design:
    - MLflow is an OPTIONAL dependency. If it's not installed, the
      helpers no-op silently so CI and lightweight environments still
      work. This mirrors how we handle shap in explain.py.
    - Runs land in mlruns/ under the project root by default (add to
      .gitignore -- these are regenerable artifacts).
    - Model registry not used here; that's a v1.1 addition once we
      have real production data to score against.

Usage:
    from src.models.tracking import mlflow_run

    with mlflow_run("xgboost_calibrated") as run:
        if run is not None:
            run.log_params({"n_estimators": 300, "max_depth": 5})
            run.log_metrics(xgb_cal_metrics)
            run.log_model(xgb_cal, name="model")
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


class _MLflowRun:
    """Thin wrapper so the caller doesn't need to import mlflow directly."""

    def __init__(self, mlflow_module) -> None:
        self._mlflow = mlflow_module

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    # Trusted types for skops serialization. MLflow 2.15+ uses skops by
    # default and rejects XGBoost/LightGBM Booster objects unless they
    # are explicitly whitelisted here.
    _TRUSTED_TYPES = [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "xgboost.sklearn.XGBRegressor",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "lightgbm.sklearn.LGBMRegressor",
    ]

    def log_model(self, model, name: str = "model") -> None:
        """Log a sklearn-compatible or XGBoost model artifact.

        Handles three moving parts across MLflow versions:
          1. `name=` kwarg (MLflow 2.9+) vs positional `artifact_path=` (older)
          2. `serialization_format='cloudpickle'` to bypass skops on 2.15+
          3. `skops_trusted_types=` as fallback whitelist if MLflow still
             routes through skops despite the cloudpickle request

        Falls back to raw pickle if everything else fails.
        """
        # Try modern signature first with EVERY safety knob turned on
        for kwargs in (
            # Modern signature (2.9+) with cloudpickle + trusted types
            dict(sk_model=model, name=name,
                 serialization_format="cloudpickle",
                 skops_trusted_types=self._TRUSTED_TYPES),
            # Modern signature with just cloudpickle (older 2.9+ MLflow)
            dict(sk_model=model, name=name,
                 serialization_format="cloudpickle"),
            # Legacy signature with cloudpickle + trusted types
            dict(sk_model=model, artifact_path=name,
                 serialization_format="cloudpickle",
                 skops_trusted_types=self._TRUSTED_TYPES),
            # Legacy signature with just cloudpickle
            dict(sk_model=model, artifact_path=name,
                 serialization_format="cloudpickle"),
        ):
            try:
                self._mlflow.sklearn.log_model(**kwargs)
                return
            except (TypeError, Exception):
                continue

        # Ultimate fallback: raw pickle as a generic artifact
        import pickle
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(model, tmp)
            tmp_path = tmp.name
        try:
            self._mlflow.log_artifact(tmp_path, artifact_path=name)
        finally:
            os.unlink(tmp_path)


def _quiet_mlflow_loggers() -> None:
    """Suppress MLflow's chatty WARNING logs about cloudpickle safety
    and pip-requirement inference fallbacks.

    Both warnings are cosmetic -- the model is still saved correctly and
    the requirements file falls back to sensible defaults -- but they
    clutter notebook output. Users who want to see them can bump the
    log level back up manually.
    """
    import logging
    for name in ("mlflow.sklearn", "mlflow.utils.environment",
                 "mlflow.models.model"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _configure_tracking_backend() -> None:
    """Configure MLflow to use a local SQLite database as the tracking
    backend. The file store is deprecated in MLflow 2.20+ and behaves
    inconsistently across versions; SQLite works reliably with the
    default `mlflow ui` command.

    Creates `mlflow.db` in the current directory the first time it runs.
    Add `mlflow.db` to .gitignore alongside `mlruns/`.
    """
    import os
    from pathlib import Path
    # Use absolute path so training runs from notebooks/ still find the
    # DB file that mlflow ui launched from the project root will read.
    db_path = (Path.cwd() / "mlflow.db").resolve()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path.as_posix()}"


@contextmanager
def mlflow_run(run_name: str,
               experiment_name: str = "streamflix_churn"
               ) -> Iterator[Optional[_MLflowRun]]:
    """Context manager that yields an MLflow run wrapper, or None if
    MLflow isn't installed.

    Callers should guard the tracking calls behind `if run is not None`
    so the pipeline works with or without MLflow.
    """
    _configure_tracking_backend()  # must run BEFORE import mlflow
    try:
        import mlflow
    except ImportError:
        yield None
        return

    _quiet_mlflow_loggers()
    # Re-set the URI in case mlflow was already imported and cached the
    # default. set_tracking_uri() takes effect immediately.
    import os
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield _MLflowRun(mlflow)