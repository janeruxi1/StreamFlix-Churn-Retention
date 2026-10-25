"""Smoke tests: the Streamlit app runs end to end and shows the reported result.

Needs streamlit, optuna and shap, which the lightweight CI job does not install,
so the module is skipped there. Run locally with `pytest tests/test_app_smoke.py`.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("optuna")
pytest.importorskip("shap")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def test_headline_matches_report(app):
    values = {m.label: m.value for m in app.metric}
    # README / memo: +$174k targeted, -$252k blanket, ROI 1.87x
    assert values["Net value / month, targeted"].startswith("$174,")
    assert values["Net value / month, blanket"].startswith("-$251,")
    assert values["ROI, targeted"] == "1.87×"


def test_every_example_subscriber_renders(app):
    for choice in app.radio(key="example").options:
        app.radio(key="example").set_value(choice).run()
        assert not app.exception, (choice, app.exception)


def test_assumptions_and_budget_update(app):
    app.slider(key="budget_k").set_value(100).run()
    app.slider(key="uplift_curated_playlist").set_value(2).run()
    assert not app.exception, app.exception
