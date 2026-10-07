from dataclasses import dataclass
import math


LABELS = {
    "anc_coverage": "ANC coverage",
    "institutional_delivery": "Institutional delivery",
    "immunization": "Immunization",
    "high_risk_cases": "High-risk cases",
}
PERCENT_COLUMNS = tuple(list(LABELS)[:3])
REQUIRED_COLUMNS = ("month", "district", *LABELS)


def label(column: str) -> str:
    return LABELS.get(column, column.replace("_", " ").capitalize())


def unit(column: str) -> str:
    return "%" if column in PERCENT_COLUMNS else ""


@dataclass(frozen=True)
class Settings:
    trend_threshold: float = 10.0
    outlier_method: str = "iqr"
    outlier_scope: str = "pooled"
    iqr_factor: float = 1.5
    z_threshold: float = 3.0
    correlation_threshold: float = 0.70
    min_outlier_rows: int = 4
    min_correlation_rows: int = 3
    medium_ratio: float = 1.5
    high_ratio: float = 2.0

    def __post_init__(self):
        for name in ("trend_threshold", "iqr_factor", "z_threshold",
                     "correlation_threshold", "medium_ratio", "high_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.correlation_threshold > 1:
            raise ValueError("correlation_threshold must be at most 1.")
        if not 1 <= self.medium_ratio < self.high_ratio:
            raise ValueError("Severity ratios must satisfy 1 <= medium < high.")
        if self.outlier_method not in {"iqr", "zscore"}:
            raise ValueError("outlier_method must be iqr or zscore.")
        if self.outlier_scope not in {"pooled", "monthly"}:
            raise ValueError("outlier_scope must be pooled or monthly.")
        if self.min_outlier_rows < 4 or self.min_correlation_rows < 3:
            raise ValueError("Use at least 4 outlier rows and 3 correlation rows.")


@dataclass(frozen=True)
class Target:
    indicator: str
    limit: float
    direction: str = "minimum"

    def __post_init__(self):
        if not math.isfinite(self.limit):
            raise ValueError("Target must be finite.")
        if self.direction not in {"minimum", "maximum"}:
            raise ValueError("Target direction must be minimum or maximum.")


def severity(score: float, settings: Settings) -> str:
    """Relative review priority; not a measure of clinical risk or confidence."""
    if math.isfinite(score) and score >= settings.high_ratio:
        return "High"
    if math.isfinite(score) and score >= settings.medium_ratio:
        return "Medium"
    return "Low"
