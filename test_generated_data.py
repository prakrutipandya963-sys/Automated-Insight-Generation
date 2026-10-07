from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from generate_data import GenerationSettings, build_dataset, save_dataset
from insights.config import PERCENT_COLUMNS, REQUIRED_COLUMNS
from insights.data import dataset_coverage, load_csv
from insights.pipeline import analyze, export_files


ROOT = Path(__file__).resolve().parents[1]


def test_generated_panel_meets_size_and_schema():
    frame, provenance, metadata = build_dataset()
    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert len(frame) == 144
    assert frame.district.nunique() == 12
    assert frame.groupby("district").month.nunique().eq(12).all()
    assert dataset_coverage(frame)["meets_size_recommendation"]
    assert not frame.isna().any().any()
    assert not frame.duplicated(["district", "month"]).any()
    assert frame[list(PERCENT_COLUMNS)].ge(0).all().all()
    assert frame[list(PERCENT_COLUMNS)].le(100).all().all()
    assert frame.high_risk_cases.ge(0).all() and frame.high_risk_cases.mod(1).eq(0).all()
    assert provenance.source.value_counts().to_dict() == {"synthetic": 132, "assignment_sample": 12}
    assert metadata["start_month"] == "2025-09"
    assert metadata["end_month"] == "2026-08"


@pytest.mark.parametrize("seed", [42, 7])
def test_supplied_values_are_preserved_for_each_seed(seed):
    frame, _, _ = build_dataset(settings=GenerationSettings(seed=seed))
    sample = load_csv(ROOT / "data" / "healthcare.csv").frame.set_index(["district", "month"])
    actual = frame.set_index(["district", "month"]).loc[sample.index, sample.columns]
    pd.testing.assert_frame_equal(actual, sample, check_dtype=False, check_exact=True)


def test_generator_is_reproducible_and_matches_delivered_csv(tmp_path):
    first, sources, metadata = build_dataset()
    second, _, _ = build_dataset()
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    different, _, _ = build_dataset(settings=GenerationSettings(seed=7))
    assert not first.equals(different)
    path = save_dataset(first, sources, metadata, tmp_path)
    assert path.read_bytes() == (ROOT / "data" / "healthcare_synthetic.csv").read_bytes()
    assert load_csv(path).provenance["csv_sha256"] == sha256(path.read_bytes()).hexdigest()


def test_synthetic_provenance_is_kept_in_reports():
    dataset = load_csv(ROOT / "data" / "healthcare_synthetic.csv")
    result = analyze(dataset)
    report = json.loads(export_files(result)["run_metadata.json"])
    assert report["dataset_provenance"]["data_type"] == "synthetic_with_assignment_sample"
    assert report["source_coverage"]["meets_size_recommendation"]
    assert any("SYNTHETIC" in warning for warning in report["warnings"])
    assert result.insights.data_origin.eq("synthetic_with_assignment_sample").all()
    assert result.insights.explanation.str.startswith("Synthetic demonstration:").all()


def test_wrong_metadata_is_not_attached_to_changed_data():
    content = (ROOT / "data" / "healthcare_synthetic.csv").read_bytes()
    metadata = json.loads((ROOT / "data" / "healthcare_synthetic.metadata.json").read_text())
    changed = content.replace(b"81.2", b"81.3", 1)
    assert changed != content
    result = load_csv(changed, metadata=metadata)
    assert result.provenance == {}
    assert any("does not match" in warning for warning in result.warnings)


def test_recorded_scenarios_have_corresponding_calculated_findings():
    dataset = load_csv(ROOT / "data" / "healthcare_synthetic.csv")
    insights = analyze(dataset).insights
    for event in dataset.provenance["scenarios"]:
        matched = insights.loc[insights.entity.eq(event["district"])
                               & insights.period.eq(event["month"])
                               & insights.indicator.eq(event["indicator"])]
        assert not matched.empty
        assert matched.value.eq(event["final_value"]).all()


def test_original_and_filtered_data_do_not_pass_size_check():
    sample = load_csv(ROOT / "data" / "healthcare.csv")
    assert not dataset_coverage(sample.frame)["meets_size_recommendation"]
    expanded = load_csv(ROOT / "data" / "healthcare_synthetic.csv")
    result = analyze(expanded, months=["2026-08"])
    assert not result.metadata["selected_coverage"]["meets_size_recommendation"]
    assert result.metadata["source_coverage"]["meets_size_recommendation"]


def test_calendar_gaps_are_not_counted_as_consecutive_history():
    frame, _, _ = build_dataset()
    missing_month = frame.loc[~frame.month.eq(pd.Timestamp("2026-01-01"))]
    assert not dataset_coverage(missing_month)["consecutive_history"]


def test_minimum_size_generation():
    frame, _, _ = build_dataset(settings=GenerationSettings(months=3))
    assert len(frame) == 36
    assert dataset_coverage(frame)["meets_size_recommendation"]
    with pytest.raises(ValueError, match="three months"):
        build_dataset(settings=GenerationSettings(months=2))
