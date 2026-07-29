"""Single source of truth for the shipped production models.

Why this module exists:
    Before this file, the string "models/churn_model_v1.pkl" was
    hardcoded in five places (Streamlit app, Phase 4/4b/6/7 notebooks).
    A rename would break silently. This module declares -- in one
    place -- what "the production model" IS, so every consumer
    (Streamlit app, notebooks, tests, downstream jobs) loads the same
    artifact and stays in sync automatically.

The consistency chain the constants below enforce:
    1. Phase 4 trains  xgb_cal (calibrated XGBoost)
    2. Phase 4 pickles xgb_cal    -> CHURN_MODEL_PATH  (key: CHURN_MODEL_ARTIFACT_KEY)
    3. Phase 4 registers same run -> CHURN_MODEL_REGISTRY_NAME @production
    4. Streamlit + Phase 5-7 load -> CHURN_MODEL_PATH  (same key)

Same chain for the Phase 8 uplift model with UPLIFT_MODEL_* constants.

Adding a new production model (e.g., v2 propensity retrain):
    - Bump the version suffix in CHURN_MODEL_PATH (v1 -> v2)
    - Retrain in Phase 4 -- it will use the new path automatically
    - Streamlit + notebooks all pick up the change on next load
    - Old v1 pickle can stay on disk as rollback insurance;
      MLflow Registry keeps an aliasable version history separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


# =====================================================================
# v1 propensity-based churn model  (Phase 4 output; shipped in Streamlit)
# =====================================================================
CHURN_MODEL_PATH = Path("models/churn_model_v1.pkl")
CHURN_MODEL_REGISTRY_NAME = "streamflix_churn_production"
CHURN_MODEL_TYPE = "XGBoost + Platt calibration"

# The pickle at CHURN_MODEL_PATH is a dict with the following keys:
#   production_model  -- calibrated XGBoost (Streamlit + Phase 5-7 use this)
#   baseline_model    -- LR baseline (documented reference; not shipped)
#   feature_names     -- ordered list for aligning inference rows
#   metrics           -- test-set PR-AUC / ROC-AUC / Brier / log_loss
#   training_meta     -- n_train/n_calib/n_test/positive_rate
CHURN_MODEL_ARTIFACT_KEY = "production_model"

# MLflow load-by-alias URI for infra-side consumers (jobs, batch scoring)
CHURN_MODEL_MLFLOW_URI = f"models:/{CHURN_MODEL_REGISTRY_NAME}@production"


# =====================================================================
# v2 causal uplift model  (Phase 8 output; informs uplift-aware policy)
# Not currently wired into the Streamlit app -- app shows v1 only.
# =====================================================================
UPLIFT_FOCUS_LEVER = "credit_5"
UPLIFT_MODEL_PATH = Path(f"models/uplift_{UPLIFT_FOCUS_LEVER}_v1.pkl")
UPLIFT_MODEL_REGISTRY_NAME = f"streamflix_uplift_{UPLIFT_FOCUS_LEVER}"

# Pickle keys for the uplift model:
#   model         -- fitted uplift meta-learner (T-, S-, X-, or ClassTransform)
#   model_type    -- which meta-learner won
#   focus_lever   -- which intervention lever this model estimates uplift for
#   feature_names -- ordered list
#   metrics_test  -- qini_auc + retention_lift_at_{10,30}pct
UPLIFT_MODEL_ARTIFACT_KEY = "model"
UPLIFT_MODEL_MLFLOW_URI = f"models:/{UPLIFT_MODEL_REGISTRY_NAME}@production"


# =====================================================================
# Loaders -- consumers should call these instead of loading pickles
# directly, so path/key changes propagate through automatically.
# =====================================================================
def load_production_churn_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the currently-shipped churn model (Phase 4 output).

    Returns:
        (model, artifact_dict) -- the calibrated XGBoost plus the full
        pickle dict (feature_names, metrics, etc.) for callers that
        need the metadata.

    Raises:
        FileNotFoundError if the pickle doesn't exist. The error message
        points at the training notebook to regenerate it.
    """
    import pickle
    if not CHURN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{CHURN_MODEL_PATH} not found. Regenerate with:\n"
            f"    python notebooks/04_modeling.py"
        )
    with open(CHURN_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact[CHURN_MODEL_ARTIFACT_KEY], artifact


def load_production_uplift_model() -> Tuple[Any, Dict[str, Any]]:
    """Load the currently-shipped uplift model (Phase 8 output).

    Returns:
        (model, artifact_dict) -- the winning meta-learner + full pickle
        dict (feature_names, focus_lever, metrics_test).

    Raises:
        FileNotFoundError if the pickle doesn't exist.
    """
    import pickle
    if not UPLIFT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{UPLIFT_MODEL_PATH} not found. Regenerate with:\n"
            f"    python notebooks/08_uplift_modeling.py"
        )
    with open(UPLIFT_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact[UPLIFT_MODEL_ARTIFACT_KEY], artifact
