from pathlib import Path
from hashlib import sha256
import json

import streamlit as st

from insights.charts import correlation_chart, district_chart, severity_chart
from insights.config import PERCENT_COLUMNS, Settings, Target, label
from insights.data import DataError, load_csv, print_report
from insights.pipeline import analyze, export_files, report_zip


ROOT = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title="District Insights", page_icon="📊", layout="wide")
    st.caption("DISTRICT HEALTHCARE / EXPLORATORY ANALYTICS")
    st.title("District Insights")
    st.write("Find monthly changes, unusual values, and relationships. Inspect the evidence behind every insight.")

    with st.sidebar:
        st.header("Data & filters")
        example = st.selectbox("Example dataset", ["Synthetic expanded (144 rows)", "Assignment sample (12 rows)"],
                               key="example_dataset")
        upload = st.file_uploader("Upload a healthcare CSV", type=["csv"])
        st.caption("An upload overrides the selected example. All processing runs in this Python app.")
    try:
        filename = "healthcare_synthetic.csv" if example.startswith("Synthetic") else "healthcare.csv"
        source = upload.getvalue() if upload is not None else ROOT / "data" / filename
        metadata = None
        if isinstance(source, bytes):
            # Recognize an exact copy of the bundled synthetic CSV even when uploaded.
            known_metadata = json.loads((ROOT / "data" / "healthcare_synthetic.metadata.json").read_text())
            if sha256(source).hexdigest() == known_metadata.get("csv_sha256"):
                metadata = known_metadata
        dataset = load_csv(source, metadata=metadata)
    except (DataError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    print_report(dataset)
    frame = dataset.frame
    content = source if isinstance(source, bytes) else source.read_bytes()
    signature = sha256(content).hexdigest()
    if st.session_state.get("dataset_signature") != signature:
        for key in ("districts", "months", "indicators", "chart_indicator", "target_indicator", "target_limit"):
            st.session_state.pop(key, None)
        st.session_state["dataset_signature"] = signature
    if dataset.provenance.get("data_type") == "synthetic_with_assignment_sample":
        st.info(f"Synthetic demonstration dataset: {dataset.provenance['generated_rows']} generated rows "
                f"and {dataset.provenance['assignment_rows']} preserved assignment rows. "
                "District names are labels; these are not observed healthcare statistics.")
    elif upload is None and filename == "healthcare.csv":
        st.info("Original assignment sample: 12 supplied illustrative rows. Their status as real observations is unverified.")
    else:
        st.caption("Uploaded dataset: source and measurement definitions are user-supplied.")
    with st.sidebar:
        districts = st.multiselect("Districts", sorted(frame["district"].unique()),
                                   default=sorted(frame["district"].unique()), key="districts")
        month_options = sorted(frame["month"].dt.strftime("%Y-%m").unique())
        months = st.multiselect("Months", month_options, default=month_options, key="months")
        indicators = st.multiselect("Indicators", dataset.indicators, default=dataset.indicators,
                                    format_func=label, key="indicators")
        st.divider()
        st.subheader("Detection rules")
        trend_threshold = st.slider("Absolute monthly change (%)", 1.0, 100.0, 10.0, 1.0,
                                    key="trend_threshold")
        method = st.selectbox("Outlier method", ["iqr", "zscore"],
                              format_func=lambda value: {"iqr": "IQR", "zscore": "Z-score"}[value],
                              key="outlier_method")
        scope = st.selectbox("Outlier reference", ["pooled", "monthly"],
                             format_func=lambda value: {"pooled": "All selected rows", "monthly": "Same-month districts"}[value],
                             key="outlier_scope")
        iqr_factor = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1, disabled=method != "iqr", key="iqr_factor")
        z_threshold = st.slider("Absolute Z-score", 1.0, 5.0, 3.0, 0.1,
                                disabled=method != "zscore", key="z_threshold")
        correlation_threshold = st.slider("Absolute Pearson correlation", 0.1, 1.0, 0.7, 0.05,
                                          key="correlation_threshold")
        with st.expander("Sample sizes & priority"):
            min_outlier_rows = st.slider("Minimum values for outliers", 4, 30, 4, key="min_outlier_rows")
            min_correlation_rows = st.slider("Minimum paired correlation rows", 3, 30, 3, key="min_correlation_rows")
            medium_ratio = st.slider("Medium priority: rule multiple", 1.0, 3.0, 1.5, 0.1, key="medium_ratio")
            high_ratio = st.slider("High priority: rule multiple", 1.1, 5.0, 2.0, 0.1, key="high_ratio")
            st.caption("Low is below the medium multiple. These cutoffs rank findings for review; they are not learned or clinically validated.")

    if not districts or not months or not indicators:
        st.info("Select at least one district, month, and indicator to analyze.")
        st.stop()

    target = None
    with st.sidebar.expander("Optional operational target"):
        enabled = st.checkbox("Check a user-defined target", key="target_enabled")
        if enabled:
            target_indicator = st.selectbox("Target indicator", indicators, format_func=label, key="target_indicator")
            direction = st.selectbox("Target type", ["minimum", "maximum"], key="target_direction")
            values = frame[target_indicator].dropna()
            if target_indicator in PERCENT_COLUMNS:
                low, high = 0.0, 100.0
            elif values.empty:
                low, high = 0.0, 100.0
            else:
                padding = max(float(values.max() - values.min()), 1.0)
                low = min(0.0, float(values.min()) - padding)
                high = max(1.0, float(values.max()) + padding)
            default = float(values.median()) if not values.empty else (low + high) / 2
            limit = st.slider("Target value", low, high, default, key="target_limit")
            target = Target(target_indicator, limit, direction)
            st.caption("Enter a target supplied by your organization. No official target is assumed.")
    try:
        settings = Settings(trend_threshold, method, scope, iqr_factor, z_threshold,
                            correlation_threshold, min_outlier_rows, min_correlation_rows,
                            medium_ratio, high_ratio)
        result = analyze(dataset, settings, districts=districts, months=months,
                         indicators=indicators, target=target)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    first, second, third, fourth = st.columns(4)
    first.metric("Selected rows", len(result.data))
    second.metric("Districts", result.data["district"].nunique())
    third.metric("Insights", len(result.insights))
    fourth.metric("High priority", int(result.insights["severity"].eq("High").sum()))
    st.caption("Trends retain the previous calendar month from the uploaded history. "
               "Outlier references and correlations are recalculated from the selected rows. "
               "Changing filters can change findings and their IDs.")
    coverage = result.metadata["selected_coverage"]
    coverage_text = (f"Selected data: {coverage['districts']} districts; "
                     f"at least {coverage['min_months_per_district']} months per district.")
    if coverage["meets_size_recommendation"]:
        st.success(coverage_text + " Meets the size recommendation; this does not establish statistical validity.")
    else:
        st.warning(coverage_text + " Below the recommended 10 districts and 3 consecutive months per district.")
    st.warning("Exploratory findings only. Repeated district observations limit correlation reliability. "
               "Severity means review priority, not clinical risk.")

    insights_tab, charts_tab, data_tab, methods_tab = st.tabs(["Insights", "Charts", "Data quality", "Methods & limits"])
    with insights_tab:
        st.subheader("Findings to review")
        if result.insights.empty:
            st.info("No findings meet the current rules. Try another selection or adjust the thresholds.")
        else:
            for _, row in result.insights.head(3).iterrows():
                with st.container(border=True):
                    st.caption(f"{row['insight_id']} / {row['type'].replace('_', ' ').upper()} / {row['severity']} priority")
                    st.write(row["explanation"])
            st.dataframe(result.insights, hide_index=True, width="stretch")
        st.caption("The same observation can have both a trend insight and an outlier insight; these test different questions.")
        files = export_files(result)
        left, middle, right = st.columns(3)
        left.download_button("Download insights CSV", files["insights.csv"], "insights.csv", "text/csv")
        middle.download_button("Download insights JSON", files["insights.json"], "insights.json", "application/json")
        right.download_button("Download full report", report_zip(result), "analytics-report.zip", "application/zip")

    with charts_tab:
        left, right = st.columns([1, 1.6])
        with left:
            st.subheader("Review priority")
            st.plotly_chart(severity_chart(result.insights), width="stretch")
        with right:
            st.subheader("Pearson correlation")
            st.plotly_chart(correlation_chart(result.correlations, result.pair_counts), width="stretch")
            st.caption("Blank cells are undefined or have too few paired values. Hover to see the paired row count.")
            st.download_button("Download correlation matrix", files["correlation_matrix.csv"],
                               "correlation_matrix.csv", "text/csv")
        st.subheader("Monthly values by district")
        chart_indicator = st.selectbox("Chart indicator", indicators, format_func=label, key="chart_indicator")
        st.plotly_chart(district_chart(result.data, chart_indicator), width="stretch")
        st.caption("Gaps stay visible. One selected month produces points rather than a trend line.")

    with data_tab:
        st.subheader("Selected data")
        st.dataframe(result.data, hide_index=True, width="stretch")
        st.subheader("Missing values in the uploaded data")
        st.dataframe(dataset.missing.rename("missing_values"), width="stretch")
        with st.expander("head() and info() for the validated upload"):
            st.dataframe(frame.head(), hide_index=True)
            st.code(dataset.info, language="text")
        with st.expander("Trend calculations, including skipped comparisons"):
            st.dataframe(result.trends, hide_index=True, width="stretch")
        with st.expander("Outlier evidence"):
            st.dataframe(result.outliers, hide_index=True, width="stretch")
        with st.expander("Correlation pair counts"):
            st.dataframe(result.pair_counts, width="stretch")

    with methods_tab:
        st.subheader("What the rules mean")
        st.markdown("""
- **Monthly change:** `(current - previous) / previous * 100`. A flag means the absolute change meets your threshold. Missing months and zero baselines are not assigned a percentage.
- **IQR:** flag values strictly outside `Q1 - k * IQR` and `Q3 + k * IQR`. Quartiles use linear interpolation.
- **Z-score:** an alternative using `(value - mean) / std`, with `ddof=0`. Flag when `abs(z) > threshold`. Constant groups have no Z-score outliers.
- **Correlation:** Pearson correlation on each pair's complete selected rows; each pair appears once. Two observations are insufficient for a useful correlation, so the minimum is three.
- **Priority:** trend score is `abs(change_pct) / threshold`; IQR score is distance from median divided by distance to the relevant fence; Z-score is `abs(z) / threshold`; correlation is `abs(r) / threshold`. Medium and High use the sidebar multiples. These scores rank rule exceedance, not confidence, and are not calibrated across methods.
- **Targets:** disabled by default. Score is `1 + abs(value - target) / abs(target)`. A zero target or zero IQR scoring distance gets an undefined score and Low priority for manual review.
""")
        for note in result.warnings:
            st.write(f"• {note}")
        with st.expander("Reproducible run settings"):
            st.json(result.metadata)


if __name__ == "__main__":
    main()
