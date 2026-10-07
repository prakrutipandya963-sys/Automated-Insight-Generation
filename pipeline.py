from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .analysis import find_correlations, find_outliers, find_trends
from .config import Settings, Target
from .data import Dataset, dataset_coverage
from .narratives import build_insights


@dataclass
class Results:
    data: pd.DataFrame
    trends: pd.DataFrame
    outliers: pd.DataFrame
    correlations: pd.DataFrame
    pair_counts: pd.DataFrame
    correlation_pairs: pd.DataFrame
    insights: pd.DataFrame
    warnings: list[str]
    metadata: dict


def analyze(dataset: Dataset, settings: Settings | None = None, *, districts=None,
            months=None, indicators=None, target: Target | None = None) -> Results:
    settings = settings or Settings()
    frame = dataset.frame
    districts = sorted(frame["district"].unique()) if districts is None else list(districts)
    months = sorted(frame["month"].dt.strftime("%Y-%m").unique()) if months is None else list(months)
    indicators = dataset.indicators if indicators is None else list(indicators)
    if not districts or not months or not indicators:
        raise ValueError("Select at least one district, month, and indicator.")
    if len(set(indicators)) != len(indicators) or not set(indicators).issubset(dataset.indicators):
        raise ValueError("Select unique indicators from the dataset.")
    if not set(districts).issubset(frame["district"].unique()):
        raise ValueError("Unknown district selected.")
    if not set(months).issubset(frame["month"].dt.strftime("%Y-%m").unique()):
        raise ValueError("Unknown month selected; use YYYY-MM.")
    if target is not None and target.indicator not in indicators:
        raise ValueError("The target indicator must be included in the selected indicators.")

    history = frame.loc[frame["district"].isin(districts), ["district", "month", *indicators]]
    data = history.loc[history["month"].dt.strftime("%Y-%m").isin(months)].copy()
    if data.empty:
        raise ValueError("There are no rows for this district/month selection.")
    # Compute before filtering the displayed months, so August can still use July.
    trends = find_trends(history, indicators, settings)
    trends = trends.loc[trends["month"].dt.strftime("%Y-%m").isin(months)].reset_index(drop=True)
    outliers, notes = find_outliers(data, indicators, settings)
    matrix, counts, pairs = find_correlations(data, indicators, settings)
    insights = build_insights(trends, outliers, pairs, data, settings, target)
    origin = dataset.provenance.get("data_type", "unspecified")
    insights["data_origin"] = origin
    if origin == "synthetic_with_assignment_sample" and not insights.empty:
        insights["explanation"] = "Synthetic demonstration: " + insights["explanation"]

    warnings = list(dataset.warnings) + notes
    if history.groupby("district")["month"].nunique().lt(3).any():
        warnings.append("Some districts have fewer than three months of history. "
                        "Month-to-month changes are available, but sustained trends cannot be established.")
    if data["district"].nunique() < 10:
        warnings.append("Fewer than 10 districts are selected. Correlations are exploratory and fragile.")
    warnings.append("Pooled correlations mix between-district and time variation; repeated rows from a "
                    "district are not independent. Correlation does not establish causation.")
    warnings.append("Severity is relative review priority, not clinical risk or statistical confidence. "
                    "high_risk_cases is a raw count; exposure denominators are unavailable.")
    skipped = trends["status"].value_counts()
    for reason in ("month_gap", "missing_value", "zero_baseline"):
        if skipped.get(reason, 0):
            warnings.append(f"Skipped {skipped[reason]} trend comparison(s): {reason.replace('_', ' ')}.")
    undefined = [column for column in indicators if pd.isna(matrix.loc[column, column])]
    if undefined:
        warnings.append("Correlation unavailable for constant or insufficiently observed indicators: "
                        + ", ".join(undefined) + ". Undefined cells are left blank, not replaced by zero.")

    metadata = {
        "settings": asdict(settings), "target": asdict(target) if target else None,
        "selection": {"districts": sorted(districts), "months": sorted(months), "indicators": indicators},
        "source_rows": len(frame), "selected_rows": len(data), "insight_count": len(insights),
        "validated_data_sha256": sha256(frame.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest(),
        "trend_baseline": "Previous calendar month from the full uploaded history of selected districts",
        "outlier_reference": settings.outlier_scope,
        "correlation_reference": "Selected district-month rows, pairwise complete observations",
        "warnings": list(dict.fromkeys(warnings)),
        "dataset_provenance": dataset.provenance,
        "source_coverage": dataset_coverage(frame),
        "selected_coverage": dataset_coverage(data),
    }
    return Results(data, trends, outliers, matrix, counts, pairs, insights, metadata["warnings"], metadata)


def export_files(result: Results) -> dict[str, str]:
    return {
        "insights.csv": result.insights.to_csv(index=False),
        "insights.json": result.insights.to_json(orient="records", indent=2, double_precision=10),
        "correlation_matrix.csv": result.correlations.to_csv(index_label="indicator"),
        "correlation_pair_counts.csv": result.pair_counts.to_csv(index_label="indicator"),
        "correlation_pairs.csv": result.correlation_pairs.to_csv(index=False),
        "trends.csv": result.trends.to_csv(index=False, date_format="%Y-%m"),
        "outliers.csv": result.outliers.to_csv(index=False, date_format="%Y-%m"),
        "selected_data.csv": result.data.to_csv(index=False, date_format="%Y-%m"),
        "run_metadata.json": json.dumps(result.metadata, indent=2, allow_nan=False),
    }


def save_results(result: Results, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in export_files(result).items():
        (folder / name).write_text(content, encoding="utf-8")


def report_zip(result: Results) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in export_files(result).items():
            archive.writestr(name, content)
    return buffer.getvalue()
