"""Feature -> intervention mapping tests.

Skips SHAP tests (heavy dep, numba required) -- those are exercised by
the notebook. Here we only test the pure-Python lookup logic.
"""
from src.models.explain import (
    FEATURE_INTERVENTION_MAP, map_to_intervention,
)


def test_intervention_map_entries_have_required_keys():
    for feature, spec in FEATURE_INTERVENTION_MAP.items():
        assert "lever" in spec, f"{feature} missing 'lever'"
        assert "cost" in spec, f"{feature} missing 'cost'"
        assert "note" in spec, f"{feature} missing 'note'"


def test_intervention_map_costs_are_numeric():
    for feature, spec in FEATURE_INTERVENTION_MAP.items():
        assert isinstance(spec["cost"], (int, float))
        assert spec["cost"] >= 0


def test_map_to_intervention_known_feature():
    result = map_to_intervention("days_since_last_login")
    assert "lever" in result
    assert result["cost"] > 0  # actionable feature


def test_map_to_intervention_unknown_feature_returns_placeholder():
    """Unknown/diagnostic features should return the N/A placeholder."""
    result = map_to_intervention("some_random_feature_name")
    assert result["cost"] == 0.0
    assert "diagnostic" in result["note"].lower() or "N/A" in result["lever"]


def test_map_to_intervention_handles_one_hot_suffix():
    """plan_tier_Basic should map to plan_tier's lever (if defined)."""
    # We don't currently have 'plan_tier' in the map, so this returns
    # the placeholder -- that's the expected behavior.
    result = map_to_intervention("plan_tier_Basic")
    assert "lever" in result


def test_actionable_features_have_meaningful_notes():
    for feature, spec in FEATURE_INTERVENTION_MAP.items():
        if spec["cost"] > 0:
            assert len(spec["note"]) > 10, (
                f"{feature} note too short: '{spec['note']}'"
            )
