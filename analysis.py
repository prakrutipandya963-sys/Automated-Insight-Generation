from itertools import combinations

import numpy as np
import pandas as pd

from .config import Settings


def find_trends(frame: pd.DataFrame, indicators: list[str], settings: Settings) -> pd.DataFrame:
    # Building one part per indicator also supports source columns named "value".
    parts = [frame[["district", "month", column]].rename(columns={column: "value"})
             .assign(indicator=column) for column in indicators]
    rows = pd.concat(parts, ignore_index=True)
    rows = rows.sort_values(["district", "indicator", "month"]).reset_index(drop=True)
    groups = rows.groupby(["district", "indicator"], sort=False)
    rows["prev_value"] = groups["value"].shift()
    rows["prev_month"] = groups["month"].shift()
    current_month = rows["month"].dt.year * 12 + rows["month"].dt.month
    prior_month = rows["prev_month"].dt.year * 12 + rows["prev_month"].dt.month
    adjacent = (current_month - prior_month).eq(1)
    rows["change"] = (rows["value"] - rows["prev_value"]).where(adjacent)
    valid = adjacent & rows["value"].notna() & rows["prev_value"].notna() & rows["prev_value"].ne(0)
    # Explicit division prevents implicit filling and undefined zero-baseline percentages.
    rows["change_pct"] = (rows["change"] / rows["prev_value"].where(valid) * 100).where(valid)
    rows["status"] = np.select(
        [rows["prev_month"].isna(), ~adjacent,
         rows["value"].isna() | rows["prev_value"].isna(), rows["prev_value"].eq(0)],
        ["no_previous_month", "month_gap", "missing_value", "zero_baseline"],
        default="ok",
    )
    rows["is_significant"] = rows["change_pct"].abs().ge(settings.trend_threshold)
    rows["score"] = rows["change_pct"].abs() / settings.trend_threshold
    return rows


OUTLIER_COLUMNS = ["district", "month", "indicator", "value", "mean", "median", "q1", "q3",
                   "lower_bound", "upper_bound", "z_score", "score", "n", "scope", "method"]


def find_outliers(frame: pd.DataFrame, indicators: list[str], settings: Settings):
    results, warnings = [], []
    groups = [("selected months", frame)] if settings.outlier_scope == "pooled" else frame.groupby("month")
    for period, group in groups:
        scope = "selected district-month rows" if settings.outlier_scope == "pooled" else f"selected districts in {period:%Y-%m}"
        for column in indicators:
            values = group[column].dropna()
            if len(values) < settings.min_outlier_rows:
                warnings.append(f"Outliers skipped for {column} ({scope}): {len(values)} valid values; "
                                f"at least {settings.min_outlier_rows} required.")
                continue
            mean, median = float(values.mean()), float(values.median())
            q1, q3 = values.quantile([0.25, 0.75], interpolation="linear")
            spread = q3 - q1
            std = float(values.std(ddof=0))
            if settings.outlier_method == "iqr":
                lower = q1 - settings.iqr_factor * spread
                upper = q3 + settings.iqr_factor * spread
            else:
                if std == 0:
                    continue
                lower = mean - settings.z_threshold * std
                upper = mean + settings.z_threshold * std
            flagged = group.loc[group[column].lt(lower) | group[column].gt(upper)]
            if settings.outlier_method == "iqr" and spread == 0 and not flagged.empty:
                warnings.append(f"{column} ({scope}) has zero IQR. Deviations are flagged by the literal "
                                "fence rule; a zero scoring distance gets Low priority for manual review.")
            for row in flagged.itertuples(index=False, name=None):
                record = dict(zip(group.columns, row))
                value = record[column]
                z_score = (value - mean) / std if std else np.nan
                if settings.outlier_method == "iqr":
                    distance = (median - lower) if value < lower else (upper - median)
                    score = abs(value - median) / distance if distance > 0 else np.nan
                else:
                    score = abs(z_score) / settings.z_threshold
                results.append(dict(district=record["district"], month=record["month"], indicator=column,
                                    value=value, mean=mean, median=median, q1=q1, q3=q3,
                                    lower_bound=lower, upper_bound=upper, z_score=z_score,
                                    score=score, n=len(values), scope=scope, method=settings.outlier_method))
    return pd.DataFrame(results, columns=OUTLIER_COLUMNS), warnings


PAIR_COLUMNS = ["indicator_a", "indicator_b", "r", "n", "districts", "months", "period", "score"]


def find_correlations(frame: pd.DataFrame, indicators: list[str], settings: Settings):
    numeric = frame[indicators]
    matrix = numeric.corr(method="pearson", min_periods=settings.min_correlation_rows)
    present = numeric.notna().astype(int)
    counts = present.T.dot(present)
    pairs = []
    for first, second in combinations(indicators, 2):
        r = matrix.loc[first, second]
        if pd.isna(r) or abs(r) < settings.correlation_threshold:
            continue
        complete = frame.dropna(subset=[first, second])
        periods = sorted(complete["month"].dt.strftime("%Y-%m").unique())
        pairs.append(dict(indicator_a=first, indicator_b=second, r=float(r), n=len(complete),
                          districts=complete["district"].nunique(), months=complete["month"].nunique(),
                          period=", ".join(periods), score=abs(r) / settings.correlation_threshold))
    return matrix, counts, pd.DataFrame(pairs, columns=PAIR_COLUMNS)
