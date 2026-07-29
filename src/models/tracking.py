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
    - Runs land in the SQLite backend at `mlflow.db` under the project
      root (add to .gitignore -- these are regenerable artifacts).
    - Model Registry IS used for production winners. After the notebooks
      pick a winner, call `register_production_model()` to promote it
      under a stable registered name with a @production alias.

Usage:
    from src.models.tracking import mlflow_run, register_production_model

    with mlflow_run("xgboost_calibrated") as run:
        if run is not None:
            run.log_params({"n_estimators": 300, "max_depth": 5})
            run.log_metrics(xgb_cal_metrics)
            run.log_model(xgb_cal, name="model")
            winner_run_id = run.run_id

    # After picking the winner, promote it to the Registry:
    register_production_model(winner_run_id, "streamflix_churn_production")

    # Later, in the app or downstream job, load by registered name:
    #   model = mlflow.pyfunc.load_model(
    #       "models:/streamflix_churn_production@production"
    #   )
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


class _MLflowRun:
    """Thin wrapper so the caller doesn't need to import mlflow directly."""

    def __init__(self, mlflow_module) -> None:
        self._mlflow = mlflow_module

    @property
    def run_id(self) -> Optional[str]:
        """The MLflow run ID -- capture this WHILE the run is active so
        it can be used later (outside the `with` block) to register the
        model in the MLflow Model Registry."""
        run = self._mlflow.active_run()
        return run.info.run_id if run is not None else None

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    # Trusted types for skops serialization. MLflow 2.15+ uses skops by
    # default and rejects XGBoost Booster objects unless they are
    # explicitly whitelisted here.
    _TRUSTED_TYPES = [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "xgboost.sklearn.XGBRegressor",
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


def register_production_model(
    run_id: str,
    registered_model_name: str,
    artifact_name: str = "model",
) -> None:
    """Promote a logged run's model artifact into the MLflow Model
    Registry as the new production version.

    Creates a new version each call, then tags it with the
    `@production` alias (MLflow 2.9+ modern API). Falls back to the
    legacy `Production` stage on older MLflow versions.

    Idempotent-ish: each call creates a new version, and the alias is
    moved to point at that version -- so calling this twice from the
    same run cleanly bumps v1 -> v2 while keeping `@production` pointing
    at the latest.

    No-ops silently if MLflow isn't installed or the run_id is None,
    so the pipeline degrades gracefully in lightweight environments.

    Args:
        run_id: The run whose model artifact should be promoted. Get
            this from `run.run_id` INSIDE the `with mlflow_run(...)`
            block (it returns None once the block exits).
        registered_model_name: Logical name in the Registry. Convention:
            `streamflix_<domain>_<purpose>`, e.g. `streamflix_churn_production`.
        artifact_name: Sub-path of the model artifact within the run
            (matches the `name=` used in `log_model`). Default "model".

    Loading the registered version later:
        model = mlflow.pyfunc.load_model(
            f"models:/{registered_model_name}@production"
        )
    """
    if run_id is None:
        return
    _configure_tracking_backend()
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return

    _quiet_mlflow_loggers()
    import os
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    model_uri = f"runs:/{run_id}/{artifact_name}"
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        version = result.version
    except Exception as e:
        print(f"  Registry: could not register {registered_model_name} ({type(e).__name__}: {e})")
        return

    client = MlflowClient()

    # Prefer the modern alias-based API (MLflow 2.9+)
    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="production",
            version=version,
        )
        print(f"  Registry: {registered_model_name} v{version} tagged @production")
        return
    except Exception:
        # Fall through to legacy stage-based API
        pass

    # Legacy stage-based API (deprecated in MLflow 2.9+ but still works)
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"  Registry: {registered_model_name} v{version} @ Production stage")
    except Exception as e:
        print(f"  Registry: stage transition failed for {registered_model_name} v{version} ({e})")


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