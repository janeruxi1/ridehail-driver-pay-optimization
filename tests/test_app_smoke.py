"""The Streamlit demo renders from the committed artifacts (skipped where streamlit is not installed)."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_app_runs_without_errors(app):
    assert not app.exception


def test_example_trip_sets_the_quote_inputs(app):
    app.selectbox(key="example").set_value("Morning commute in Manhattan").run()
    assert not app.exception
    assert app.selectbox(key="q_pu").value == 161 and app.number_input(key="q_miles").value == 2.4


def test_rollout_power_reacts_to_weeks(app):
    before = [m.value for m in app.metric]
    app.slider[-1].set_value(2).run()
    assert not app.exception
    assert [m.value for m in app.metric] != before
