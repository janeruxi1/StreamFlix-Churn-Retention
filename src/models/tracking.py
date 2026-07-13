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

    def log_model(self, model, name: str = "model") -> None:
        """Log a sklearn-compatible or XGBoost model artifact."""
        try:
            self._mlflow.sklearn.log_model(model, name)
        except Exception:
            # Fall back to a generic pickle if sklearn.log_model can't handle it
            import pickle
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(
                suffix=".pkl", delete=False
            ) as tmp:
                pickle.dump(model, tmp)
                tmp_path = tmp.name
            try:
                self._mlflow.log_artifact(tmp_path, artifact_path=name)
            finally:
                os.unlink(tmp_path)


@contextmanager
def mlflow_run(run_name: str,
               experiment_name: str = "streamflix_churn"
               ) -> Iterator[Optional[_MLflowRun]]:
    """Context manager that yields an MLflow run wrapper, or None if
    MLflow isn't installed.

    Callers should guard the tracking calls behind `if run is not None`
    so the pipeline works with or without MLflow.
    """
    try:
        import mlflow
    except ImportError:
        yield None
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield _MLflowRun(mlflow)
