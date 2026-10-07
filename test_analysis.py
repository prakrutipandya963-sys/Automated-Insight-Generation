import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from insights.analysis import find_correlations, find_outliers, find_trends
from insights.config import Settings, Target, severity
from insights.data import DataError, load_csv, validate_data
from insights.pipeline import analyze, export_files, report_zip


SAMPLE = Path(__file__).resolve().parents[1] / "data" / "healthcare.csv"


@pytest.fixture
def dataset():
    return load_csv(SAMPLE)


def test_sample_expected_findings(dataset):
    result = analyze(dataset)
    assert result.insights["type"].value_counts().to_dict() == {"trend": 6, "outlier": 4, "correlation": 2}
    anc = result.trends.query("district == 'Ahmedabad' and indicator == 'anc_coverage' and status == 'ok'").iloc[0]
    assert anc.change_pct == pytest.approx(-16 / 85 * 100)
    assert anc.change == -16
    assert bool(anc.is_significant)
    mehsana = result.outliers.query("district == 'Mehsana'")
    assert set(mehsana.indicator) == {"anc_coverage", "high_risk_cases"}
    anc_outlier = mehsana.query("indicator == 'anc_coverage'").iloc[0]
    # Sorted ANC values give Q1=78.5 and Q3=84.25 by linear interpolation.
    assert anc_outlier.lower_bound == 69.875
    assert anc_outlier["mean"] == pytest.approx(940 / 12)
    assert result.correlations.loc["anc_coverage", "high_risk_cases"] == pytest.approx(-0.933495, abs=1e-6)


def test_names_are_data_driven(dataset):
    changed = dataset.frame.copy()
    changed["district"] = changed["district"].replace({"Mehsana": "New District"})
    rows = analyze(validate_data(changed)).insights
    assert not rows.explanation.str.contains("Mehsana").any()
    assert rows.explanation.str.contains("New District").any()


def test_values_are_data_driven(dataset):
    changed = dataset.frame.copy()
    mask = changed.district.eq("Ahmedabad") & changed.month.eq(pd.Timestamp("2026-08-01"))
    changed.loc[mask, "anc_coverage"] = 85
    rows = analyze(validate_data(changed)).insights
    assert rows.query("entity == 'Ahmedabad' and indicator == 'anc_coverage' and type == 'trend'").empty


def test_latest_month_keeps_previous_month(dataset):
    result = analyze(dataset, months=["2026-08"], districts=["Ahmedabad"], indicators=["anc_coverage"])
    assert len(result.data) == 1
    assert result.trends.iloc[0].prev_value == 85
    assert result.trends.iloc[0].change_pct == pytest.approx(-16 / 85 * 100)
    assert result.correlations.isna().all().all()


@pytest.mark.parametrize("reason,prior,current,months", [
    ("zero_baseline", 0, 10, ["2026-07", "2026-08"]),
    ("zero_baseline", 0, 0, ["2026-07", "2026-08"]),
    ("missing_value", np.nan, 10, ["2026-07", "2026-08"]),
    ("missing_value", 10, np.nan, ["2026-07", "2026-08"]),
    ("month_gap", 10, 20, ["2026-07", "2026-09"]),
])
def test_undefined_monthly_changes(reason, prior, current, months):
    frame = pd.DataFrame({"district": ["A", "A"], "month": pd.to_datetime(months), "value": [prior, current]})
    row = find_trends(frame, ["value"], Settings()).iloc[1]
    assert row.status == reason
    assert pd.isna(row.change_pct)
    assert not row.is_significant


def test_exact_trend_threshold_and_unsorted_dates():
    frame = pd.DataFrame({"district": ["A", "A"], "month": pd.to_datetime(["2026-08", "2026-07"]),
                          "value": [110, 100]})
    row = find_trends(frame, ["value"], Settings()).iloc[1]
    assert row.change_pct == 10
    assert row.is_significant


def test_monthly_and_pooled_reference_differ(dataset):
    pooled = analyze(dataset)
    monthly = analyze(dataset, Settings(outlier_scope="monthly"))
    assert pooled.outliers["n"].eq(12).all()
    assert monthly.outliers["n"].eq(6).all()
    row = monthly.outliers.query("district == 'Mehsana' and indicator == 'anc_coverage'").iloc[0]
    assert row["mean"] == 74
    assert row.lower_bound == 55.375


def test_zero_iqr_preserves_literal_fence_rule():
    frame = pd.DataFrame({"district": list("ABCDE"), "month": pd.to_datetime(["2026-08"] * 5),
                          "value": [5, 5, 5, 5, 10]})
    rows, notes = find_outliers(frame, ["value"], Settings())
    assert rows.value.tolist() == [10]
    assert pd.isna(rows.iloc[0].score)
    assert notes
    assert severity(rows.iloc[0].score, Settings()) == "Low"


def test_constant_values_have_no_outliers_or_correlation():
    frame = pd.DataFrame({"district": list("ABCD"), "month": pd.to_datetime(["2026-08"] * 4),
                          "one": [5] * 4, "two": [6] * 4})
    for method in ["iqr", "zscore"]:
        outliers, _ = find_outliers(frame, ["one", "two"], Settings(outlier_method=method))
        assert outliers.empty
    matrix, _, pairs = find_correlations(frame, ["one", "two"], Settings())
    assert matrix.isna().all().all()
    assert pairs.empty


def test_zscore_uses_population_std(dataset):
    assert analyze(dataset, Settings(outlier_method="zscore")).outliers.empty
    result = analyze(dataset, Settings(outlier_method="zscore", z_threshold=2.5))
    row = result.outliers.query("indicator == 'high_risk_cases'").iloc[0]
    values = dataset.frame.high_risk_cases.to_numpy()
    expected_z = (28 - np.mean(values)) / np.std(values, ddof=0)
    assert row.z_score == pytest.approx(expected_z)
    assert row.z_score > 2.5


def test_iqr_fences_are_strict():
    # Q1=2.5, Q3=5.5, upper fence=10 exactly.
    frame = pd.DataFrame({"district": list("ABCDEFG"), "month": pd.to_datetime(["2026-08"] * 7),
                          "value": [1, 2, 3, 4, 5, 6, 10]})
    rows, _ = find_outliers(frame, ["value"], Settings())
    assert rows.empty


def test_pairwise_missing_counts_and_no_two_point_correlation():
    frame = pd.DataFrame({"district": list("ABCD"), "month": pd.to_datetime(["2026-08"] * 4),
                          "a": [1, 2, 3, 4], "b": [2, np.nan, np.nan, 8], "c": [2, 4, 6, np.nan]})
    matrix, counts, pairs = find_correlations(frame, ["a", "b", "c"], Settings())
    assert counts.loc["a", "b"] == 2
    assert pd.isna(matrix.loc["a", "b"])
    assert matrix.loc["a", "c"] == pytest.approx(1.0)
    assert pairs[["indicator_a", "indicator_b"]].values.tolist() == [["a", "c"]]


def test_severity_tracks_user_threshold(dataset):
    default = analyze(dataset).insights
    higher = analyze(dataset, Settings(trend_threshold=15)).insights
    select = "entity == 'Ahmedabad' and indicator == 'anc_coverage' and type == 'trend'"
    assert default.query(select).iloc[0].severity == "Medium"
    assert higher.query(select).iloc[0].severity == "Low"
    assert analyze(dataset, Settings(trend_threshold=20)).insights.query(select).empty


def test_optional_target_has_no_invented_default(dataset):
    assert not analyze(dataset).insights.type.eq("threshold_breach").any()
    result = analyze(dataset, target=Target("anc_coverage", 80))
    breaches = result.insights.query("type == 'threshold_breach'")
    assert len(breaches) == 5
    assert breaches.value.lt(80).all()
    assert breaches.explanation.str.contains("user-defined").all()


def test_zero_target_and_empty_findings(dataset):
    result = analyze(dataset, target=Target("high_risk_cases", 0, "maximum"))
    breaches = result.insights.query("type == 'threshold_breach'")
    assert breaches.score.isna().all()
    assert breaches.severity.eq("Low").all()
    result = analyze(dataset, districts=["Surat"], months=["2026-07"], indicators=["anc_coverage"])
    assert result.insights.empty
    assert "explanation" in result.insights.columns
    assert json.loads(export_files(result)["insights.json"]) == []


def test_exports_are_valid_and_reproducible(dataset):
    first = analyze(dataset)
    second = analyze(validate_data(dataset.frame.sample(frac=1, random_state=7)))
    assert export_files(first) == export_files(second)
    files = export_files(first)
    def reject(value):
        raise AssertionError(f"Invalid JSON number: {value}")
    rows = json.loads(files["insights.json"], parse_constant=reject)
    assert len(rows) == 12
    assert rows[0]["insight_id"] == "INS-0001"
    correlation = next(row for row in rows if row["type"] == "correlation")
    assert isinstance(correlation["metric"], float)
    assert correlation["change"] is None
    with ZipFile(BytesIO(report_zip(first))) as archive:
        assert set(archive.namelist()) == set(files)


def test_extra_numeric_indicator(dataset):
    frame = dataset.frame.copy()
    frame["clinic_visits"] = range(100, 112)
    added = validate_data(frame)
    result = analyze(added)
    assert "clinic_visits" in added.indicators
    assert "clinic_visits" in result.correlations.columns


@pytest.mark.parametrize("column,value,message", [
    ("anc_coverage", 101, "between 0 and 100"),
    ("anc_coverage", -1, "between 0 and 100"),
    ("immunization", "bad", "non-numeric"),
    ("high_risk_cases", -1, "whole counts"),
    ("high_risk_cases", 1.5, "whole counts"),
    ("high_risk_cases", np.inf, "infinite"),
    ("month", "2026-13", "invalid calendar"),
    ("month", "2026-02-30", "invalid calendar"),
    ("month", "08/01/26", "YYYY-MM"),
    ("district", "  ", "cannot be missing"),
])
def test_invalid_values_are_rejected(dataset, column, value, message):
    frame = dataset.frame.astype(object)
    frame["month"] = dataset.frame.month.dt.strftime("%Y-%m")
    frame.loc[0, column] = value
    with pytest.raises(DataError, match=message):
        validate_data(frame)


def test_duplicate_keys_rejected_after_month_normalization(dataset):
    frame = dataset.frame.iloc[[0, 0]].copy()
    frame["month"] = ["2026-07", "2026-07-20"]
    with pytest.raises(DataError, match="Duplicate"):
        validate_data(frame)


@pytest.mark.parametrize("content", [b"", b"a,a\n1,2", b"month,district\n2026-07,A"])
def test_invalid_schema(content):
    with pytest.raises(DataError):
        load_csv(content)


def test_missing_values_remain_missing(dataset):
    frame = dataset.frame.copy()
    frame.loc[0, "anc_coverage"] = np.nan
    validated = validate_data(frame)
    assert validated.missing["anc_coverage"] == 1
    assert pd.isna(validated.frame.loc[0, "anc_coverage"])
    assert validated.warnings


@pytest.mark.parametrize("kwargs", [{"trend_threshold": 0}, {"correlation_threshold": 1.1},
                                    {"iqr_factor": float("nan")}, {"medium_ratio": 2, "high_ratio": 1.5}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


@pytest.mark.parametrize("kwargs", [{"districts": []}, {"months": []}, {"indicators": []},
                                    {"districts": ["Unknown"]}, {"months": ["2020-01"]}])
def test_invalid_selections(dataset, kwargs):
    with pytest.raises(ValueError):
        analyze(dataset, **kwargs)
