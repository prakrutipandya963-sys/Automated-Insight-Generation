from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def start_app():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    return app.selectbox(key="example_dataset").set_value("Assignment sample (12 rows)").run()


def test_dashboard_loads_all_required_outputs():
    app = start_app()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["12", "6", "12", "4"]
    assert len(app.get("plotly_chart")) == 3
    assert len(app.get("download_button")) == 4


def test_live_filters_and_trend_slider():
    app = start_app()
    app.multiselect(key="districts").set_value(["Ahmedabad"])
    app.multiselect(key="months").set_value(["2026-08"])
    app.multiselect(key="indicators").set_value(["anc_coverage"]).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[2].value == "1"
    assert any("85% to 69%" in element.value for element in app.markdown)
    app.slider(key="trend_threshold").set_value(20.0).run()
    assert not app.exception
    assert app.metric[2].value == "0"


def test_empty_selection_and_invalid_priority_are_handled():
    app = start_app()
    app.multiselect(key="districts").set_value([]).run()
    assert not app.exception
    assert any("Select at least one" in element.value for element in app.info)
    app.multiselect(key="districts").set_value(["Surat"])
    app.slider(key="medium_ratio").set_value(3.0).run()
    assert not app.exception
    assert "medium < high" in app.error[0].value


def test_alternative_outlier_rules_and_target():
    app = start_app()
    app.selectbox(key="outlier_method").set_value("zscore")
    app.selectbox(key="outlier_scope").set_value("monthly").run()
    assert not app.exception
    app.checkbox(key="target_enabled").check().run()
    assert not app.exception
    app.slider(key="target_limit").set_value(80.0).run()
    assert not app.exception
    insights = app.dataframe[0].value
    assert insights["type"].eq("threshold_breach").any()
    app.multiselect(key="indicators").set_value(["high_risk_cases"]).run()
    assert not app.exception


def test_expanded_default_and_switching_datasets():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:2]] == ["144", "12"]
    assert any("132 generated rows" in item.value for item in app.info)
    assert any("Meets the size recommendation" in item.value for item in app.success)
    assert len(app.get("plotly_chart")) == 3
    app.multiselect(key="districts").set_value(["Anand"])
    app.multiselect(key="months").set_value(["2025-09"]).run()
    assert not app.exception
    assert not app.success
    app.selectbox(key="example_dataset").set_value("Assignment sample (12 rows)").run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:2]] == ["12", "6"]
    assert set(app.multiselect(key="months").value) == {"2026-07", "2026-08"}
    app.selectbox(key="example_dataset").set_value("Synthetic expanded (144 rows)").run()
    assert not app.exception
    assert app.metric[0].value == "144"
