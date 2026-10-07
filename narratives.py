import numpy as np
import pandas as pd

from .config import PERCENT_COLUMNS, Settings, Target, label, severity, unit


COLUMNS = ["insight_id", "type", "indicator", "entity", "period", "metric", "change",
           "value", "prev_value", "change_pct", "severity", "explanation", "score",
           "method", "reference", "n"]


def base_row(kind, indicator, entity, period, value, score, settings, **details):
    row = {column: np.nan for column in COLUMNS}
    row.update(type=kind, indicator=indicator, entity=entity, period=period,
               metric=value, value=value, score=score, severity=severity(score, settings))
    row.update(details)
    return row


def trend_insights(trends: pd.DataFrame, settings: Settings) -> list[dict]:
    results = []
    for row in trends.loc[trends["is_significant"]].itertuples(index=False):
        direction = "increased" if row.change > 0 else "decreased"
        change_unit = "percentage points" if row.indicator in PERCENT_COLUMNS else "units"
        explanation = (
            f"{label(row.indicator)} in {row.district} {direction} from "
            f"{row.prev_value:g}{unit(row.indicator)} to {row.value:g}{unit(row.indicator)} "
            f"in {row.month:%Y-%m}: {row.change_pct:+.1f}% relative change "
            f"({row.change:+g} {change_unit}) from {row.prev_month:%Y-%m}. "
            f"This meets the {settings.trend_threshold:g}% absolute-change rule; "
            "it is a month-to-month change, not a statistical significance test."
        )
        results.append(base_row("trend", row.indicator, row.district, f"{row.month:%Y-%m}",
                                row.value, row.score, settings, change=row.change,
                                prev_value=row.prev_value, change_pct=row.change_pct,
                                explanation=explanation, method="month_over_month",
                                reference=f"Previous month: {row.prev_month:%Y-%m}", n=2))
    return results


def outlier_insights(outliers: pd.DataFrame, settings: Settings) -> list[dict]:
    results = []
    for row in outliers.itertuples(index=False):
        direction = "below" if row.value < row.lower_bound else "above"
        boundary = row.lower_bound if direction == "below" else row.upper_bound
        if row.method == "iqr":
            detail = (f"the {settings.iqr_factor:g} x IQR fence of {boundary:.2f} "
                      f"(Q1={row.q1:.2f}, Q3={row.q3:.2f})")
        else:
            detail = f"the Z-score boundary of {boundary:.2f} (z={row.z_score:.2f}, ddof=0)"
        explanation = (
            f"{label(row.indicator)} in {row.district} is {row.value:g}{unit(row.indicator)} "
            f"in {row.month:%Y-%m}, {direction} {detail}. Reference: {row.scope}, "
            f"n={row.n}, mean={row.mean:.2f}. This is a potential outlier for review."
        )
        results.append(base_row("outlier", row.indicator, row.district, f"{row.month:%Y-%m}",
                                row.value, row.score, settings, change=row.value - row.mean,
                                explanation=explanation, method=row.method, reference=row.scope, n=row.n))
    return results


def correlation_insights(pairs: pd.DataFrame, settings: Settings) -> list[dict]:
    results = []
    for row in pairs.itertuples(index=False):
        direction = "positive" if row.r > 0 else "negative"
        explanation = (
            f"{label(row.indicator_a)} and {label(row.indicator_b)} have a {direction} "
            f"Pearson correlation (r={row.r:.3f}) across {row.n} paired rows, "
            f"{row.districts} district(s) and {row.months} month(s). "
            f"This meets |r| >= {settings.correlation_threshold:g}. "
            "This pooled association does not establish causation or a within-district relationship."
        )
        if row.districts < 10:
            explanation += " Fewer than 10 districts: treat this estimate as fragile."
        results.append(base_row("correlation", f"{row.indicator_a}:{row.indicator_b}",
                                "Selected districts (pooled)", row.period, row.r, row.score,
                                settings, explanation=explanation, method="pearson",
                                reference="Pairwise complete selected district-month rows", n=row.n))
    return results


def target_insights(frame: pd.DataFrame, target: Target | None, settings: Settings) -> list[dict]:
    if target is None:
        return []
    values = frame[target.indicator]
    mask = values.lt(target.limit) if target.direction == "minimum" else values.gt(target.limit)
    results = []
    for _, row in frame.loc[mask].iterrows():
        value = row[target.indicator]
        change = value - target.limit
        # A score of 1 at the boundary, plus the fractional departure from the target.
        score = 1 + abs(change) / abs(target.limit) if target.limit else np.nan
        explanation = (
            f"{label(target.indicator)} in {row['district']} is {value:g}{unit(target.indicator)} "
            f"in {row['month']:%Y-%m}, breaching the user-defined {target.direction} "
            f"target of {target.limit:g}{unit(target.indicator)}. "
            "This is a user-entered operational rule, not an official clinical standard."
        )
        if target.limit == 0:
            explanation += " Relative priority is undefined for a zero target; review manually."
        results.append(base_row("threshold_breach", target.indicator, row["district"],
                                f"{row['month']:%Y-%m}", value, score, settings,
                                change=change, explanation=explanation, method="user_target",
                                reference=f"{target.direction} target: {target.limit:g}", n=1))
    return results


def build_insights(trends, outliers, pairs, frame, settings, target=None):
    rows = (trend_insights(trends, settings) + outlier_insights(outliers, settings)
            + correlation_insights(pairs, settings) + target_insights(frame, target, settings))
    result = pd.DataFrame(rows, columns=COLUMNS)
    if not result.empty:
        result["priority"] = result["severity"].map({"High": 0, "Medium": 1, "Low": 2})
        result = result.sort_values(["priority", "score", "type", "entity", "indicator", "period"],
                                    ascending=[True, False, True, True, True, True],
                                    kind="stable", na_position="last").drop(columns="priority").reset_index(drop=True)
        result["insight_id"] = [f"INS-{number:04d}" for number in range(1, len(result) + 1)]
    return result
