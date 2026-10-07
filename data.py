import csv
from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PERCENT_COLUMNS, REQUIRED_COLUMNS


class DataError(ValueError):
    """Input errors that can be shown directly to the user."""


@dataclass
class Dataset:
    frame: pd.DataFrame
    indicators: list[str]
    missing: pd.Series
    info: str
    warnings: list[str]
    provenance: dict = field(default_factory=dict)


def load_csv(source: bytes | str | Path, *, metadata: dict | None = None) -> Dataset:
    content = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        text = content.decode("utf-8-sig")
        header = [name.strip() for name in next(csv.reader(StringIO(text)))]
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DataError("Provide a non-empty UTF-8, comma-separated CSV.") from exc
    if any(not name for name in header) or len(set(header)) != len(header):
        raise DataError("Column names must be non-empty and unique.")
    try:
        frame = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataError(f"CSV could not be read: {exc}") from exc
    # Reject ragged records that pandas may otherwise interpret as a row index.
    records = csv.reader(StringIO(text))
    next(records)
    if any(len(row) != len(header) for row in records if row):
        raise DataError("Every CSV record must have the same number of fields as the header.")
    frame.columns = header
    dataset = validate_data(frame)
    if metadata is None and not isinstance(source, bytes):
        sidecar = Path(source).with_suffix(".metadata.json")
        if sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                dataset.warnings.append("Dataset metadata could not be read; provenance is unverified.")
    if metadata is not None:
        if isinstance(metadata, dict) and metadata.get("csv_sha256") == sha256(content).hexdigest():
            dataset.provenance = metadata
            if metadata.get("data_type") == "synthetic_with_assignment_sample":
                dataset.warnings.append("SYNTHETIC DEMONSTRATION DATA: generated values plus the supplied "
                                        "assignment sample. Findings illustrate software behavior, not real district performance.")
        else:
            dataset.warnings.append("Dataset metadata does not match this CSV; provenance is unverified.")
    return dataset


def dataset_coverage(frame: pd.DataFrame) -> dict:
    """Check the brief's dataset-size recommendation for the current cohort."""
    months_per_district = frame.groupby("district")["month"].nunique()
    complete = True
    for _, group in frame.groupby("district"):
        expected = pd.date_range(group.month.min(), group.month.max(), freq="MS")
        if len(group) != len(expected):
            complete = False
    count = int(frame.district.nunique())
    minimum = int(months_per_district.min()) if len(months_per_district) else 0
    return {"rows": len(frame), "districts": count, "min_months_per_district": minimum,
            "consecutive_history": complete and count > 0,
            "meets_size_recommendation": count >= 10 and minimum >= 3 and complete}


def validate_data(frame: pd.DataFrame) -> Dataset:
    frame = frame.copy().reset_index(drop=True)
    if not frame.columns.is_unique:
        raise DataError("Column names must be unique.")
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise DataError(f"Missing required columns: {', '.join(sorted(missing_columns))}.")
    if frame.empty:
        raise DataError("The CSV contains no data rows.")
    warnings = []
    for column in ("district", "month"):
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
        if frame[column].isna().any():
            raise DataError(f"{column} cannot be missing.")
    dates = frame["month"]
    if not dates.str.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?").all():
        raise DataError("Use YYYY-MM or YYYY-MM-DD dates in month.")
    dates = dates.where(dates.str.len() == 10, dates + "-01")
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise DataError("month contains an invalid calendar date.")
    if parsed.dt.day.ne(1).any():
        warnings.append("Dates were normalized to their calendar month.")
    frame["month"] = parsed.dt.to_period("M").dt.to_timestamp()
    duplicates = frame.duplicated(["district", "month"], keep=False)
    if duplicates.any():
        rows = (frame.index[duplicates] + 2).tolist()
        raise DataError(f"Duplicate district-month keys at data line(s) {rows[:10]}. "
                        "Resolve them before analysis; values are not silently averaged.")

    indicators = [column for column in frame if column not in {"district", "month"}]
    for column in indicators:
        values = frame[column].astype("string").str.strip()
        values = values.mask(values.str.lower().isin(["", "na", "n/a", "nan", "null", "none"]))
        numbers = pd.to_numeric(values, errors="coerce").astype(float)
        if (values.notna() & numbers.isna()).any():
            raise DataError(f"{column} contains non-numeric values. "
                            "Extra columns must also be numeric indicators.")
        if np.isinf(numbers).any():
            raise DataError(f"{column} contains an infinite value.")
        if column in PERCENT_COLUMNS and (numbers.dropna().lt(0) | numbers.dropna().gt(100)).any():
            raise DataError(f"{column} must be between 0 and 100; use 85 for 85%.")
        if column == "high_risk_cases" and (numbers.dropna().lt(0) | numbers.dropna().mod(1).ne(0)).any():
            raise DataError("high_risk_cases must contain non-negative whole counts.")
        frame[column] = numbers
    frame = frame.sort_values(["district", "month"]).reset_index(drop=True)
    missing = frame.isna().sum()
    if missing.sum():
        warnings.append("Missing indicator values are kept missing. No imputation is used; "
                        "each calculation uses only its available observations.")
    buffer = StringIO()
    frame.info(buf=buffer)
    return Dataset(frame, indicators, missing, buffer.getvalue(), warnings)


def print_report(dataset: Dataset) -> None:
    print("First five validated rows:")
    print(dataset.frame.head().to_string(index=False))
    print("\nDataFrame info:")
    print(dataset.info)
    print("Missing values per column:")
    print(dataset.missing.to_string())
    for warning in dataset.warnings:
        print(f"Note: {warning}")
