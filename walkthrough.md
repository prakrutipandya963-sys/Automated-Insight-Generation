# Run first, then understand the code

## A short demo to follow

1. Run `python -m streamlit run app.py` in the project environment.
2. The default is the expanded synthetic dataset: 144 rows across 12 districts,
   with a visible synthetic-data label and dataset-size check. Inspect the full
   year in Charts. Then choose **Assignment sample (12 rows)** in the sidebar;
   the following original-sample demo shows 12 insights across 6 districts.
3. Read Mehsana's high-risk count change: 11 to 28, a rise of 154.5%.
4. Read its ANC change: 84 to 42, a fall of 50%, or 42 percentage points.
5. Open Charts and inspect all three plots. Hover over the correlation heatmap
   to see that the strong pooled associations use 12 paired rows.
6. Select only August. The trend comparison still uses July, while reference
   statistics now use the six selected August rows.
7. Select only Ahmedabad and ANC coverage. Its August change remains visible;
   outlier and correlation calculations have too little data and say so.
8. Raise the monthly change threshold to 20%. The -18.8% ANC trend disappears.
9. Restore all filters and default thresholds. Switch to Z-score at 3: there
   are no outliers. This is the correct result for that method on this sample.
10. Download the full report and open `run_metadata.json` to see the exact rules.

## Read the files in this order

### 1. `data/healthcare.csv`

The original sample is unchanged. The expanded default file is
`data/healthcare_synthetic.csv`; read `data/README.md` for its assumptions and
`generate_data.py` for its reproducible generation. All 12 supplied rows are
preserved in that file; the other 132 rows are synthetic.

Each row represents one district in one calendar month. Coverage indicators are
stored as percentages from 0 to 100; high-risk cases is a count. There are no
population denominators and no prediction label.

### 2. `insights/config.py`

`Settings` stores thresholds in one place. Its validation prevents impossible
configurations such as a zero percentage threshold or a High cutoff below the
Medium cutoff. `label()` formats column names without embedding district names.
`severity()` converts a numeric rule-relative score into a review priority.

### 3. `insights/data.py`

`load_csv()` reads bytes and checks CSV structure. `validate_data()` checks the
schema, month values, duplicate keys, numeric values, percentage bounds, and
whole case counts. It preserves missing indicators and sorts the data.
`Dataset` simply holds the cleaned frame and its diagnostic information.

### 4. `insights/analysis.py`

`find_trends()` reshapes each indicator into rows, groups by district and
indicator, and uses `shift()` to get the prior observation. It checks that the
prior observation is actually the previous calendar month before dividing.

For Ahmedabad ANC:

```text
previous = 85
current = 69
change = 69 - 85 = -16 percentage points
change_pct = (-16 / 85) * 100 = -18.823529...%
score = 18.823529... / 10 = 1.8823529...
```

The sign is retained in the export. Absolute magnitude is used for flagging and
priority. A cutoff breach is not a statistical hypothesis test.

`find_outliers()` computes either quartile fences or mean/standard-deviation
boundaries. It stores the reference statistics alongside each flagged row.
For pooled ANC, Q1=78.5 and Q3=84.25, so IQR=5.75 and the lower fence is 69.875.
Both 69 and 42 lie below that fence.

`find_correlations()` calls Pandas Pearson correlation, counts complete pairs,
and visits every unique indicator pair once. A null coefficient is left null.

### 5. `insights/narratives.py`

This file turns calculated evidence into complete sentences. It does not have
an Ahmedabad branch or a Mehsana branch. Every sentence reads the district,
month, indicator, and numeric values from a calculated row.

`base_row()` supplies the shared schema. Separate small functions produce each
insight type. `build_insights()` combines them, sorts them by priority, and adds
auto-numbered IDs. The same observation may have a trend and an outlier finding.

### 6. `insights/pipeline.py`

`analyze()` applies the chosen scope, runs all calculations, builds narratives,
and records limitations. It returns one `Results` object for either interface.
The export helpers generate CSVs, standards-compliant JSON, and a ZIP report.
No numeric results are hardcoded in the export layer.

### 7. `app.py` and `run_analysis.py`

The Streamlit script gathers filters and thresholds, calls `analyze()`, and
displays its output. Streamlit reruns the script when a widget changes. The CLI
uses the same pipeline, making it suitable for reproducible reviewer checks.
`charts.py` keeps Plotly code out of the statistical calculation functions.

## Questions to be ready for

**Why didn't you train an ML model?**

The task requires transparent descriptive rules, and the supplied sample has
only 12 rows with no labeled target. Expanding it synthetically does not add
independent real observations. A trained model would add unvalidated assumptions.
The useful engineering work is reliable calculations and traceable insights.

**Why not use an LLM to write insights?**

Templates are explicitly permitted. They are deterministic, need no API key,
and keep the values tied directly to computed evidence. An LLM adds no necessary
capability for these sentence types.

**Why IQR instead of Z-score?**

IQR is specified and easy to inspect. Mean and standard deviation can be pulled
by extreme values. Both methods remain available, and their different results
are reported honestly. Neither method is claimed to be universally best.

**What is the difference between a trend and an outlier?**

A trend flag compares a district with its own previous month. An outlier flag
compares its value with a reference distribution. One observation can satisfy
both rules, but they are separate questions.

**Why isn't Ahmedabad's ANC change labeled High like the PDF?**

The brief does not define the formula for severity. This implementation uses
threshold-relative scores with visible cutoffs. Its score is 1.882, below the
default High cutoff of 2. A district-specific override would be hardcoding.

**What does a strong correlation prove?**

Only that the selected paired values have a strong linear association. Twelve
rows contain repeated observations from six districts, and pooled relationships
can reflect district differences. They do not establish causation.

**How do you handle missing values and zeros?**

Missing values stay missing. Each method uses the observations it needs and
reports its sample size. A zero previous value has no defined relative percent
change, so it is recorded with a skip reason instead of infinity.

**How do you know the engine is general-purpose?**

The analysis iterates over observed districts and selected indicators. Tests
rename districts, edit values, add indicators, and shuffle rows. Narratives
change with the data and are stable for equivalent input.

**How would you improve it with more data?**

First collect more months, more districts, denominator information, and reviewed
findings. Then assess seasonal baselines, robust anomaly models, and separate
within-district relationships. Validate any additions against useful analyst
outcomes instead of optimizing for the assignment's 12 observations.

Before presenting, run the demo yourself and make sure you can follow the
calculation behind each finding. Explain the implemented choices and observed
results in your own words.

**Why use synthetic data, and how realistic is it?**

No larger dataset was provided. The generator preserves the given sample and
creates a clearly labeled panel with 12 districts and 12 months each to exercise
the engine beyond the minimum size. The values have valid bounds, modest monthly
coverage variability, district differences, integer counts, and disclosed test
events. Its parameters are engineering assumptions inspired by the sample scale,
not estimates validated against actual district healthcare records.

**Why are there so many count-change flags?**

The assignment asks for a 10% rule. On small counts, a difference of one or two
cases can meet that threshold. Poisson-generated counts include ordinary random
variation; flagged changes are not automatically meaningful anomalies. The rule
is preserved and its limitations are reported instead of suppressing inconvenient
findings. Stakeholders could later specify per-indicator and absolute-change
criteria, but those are not silently added to this assignment's default rule.

**Does the larger dataset prove better correlations?**

No. It meets the numerical size recommendation for a demonstration. Synthetic
correlations reflect generation assumptions and repeated district measurements.
Real-world validity still requires appropriate observed data and review.
