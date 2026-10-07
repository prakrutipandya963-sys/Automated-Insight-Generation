# Design decisions

## 1. VS Code and a normal Python project

The deliverable is an interactive dashboard with several reusable functions,
tests, and exports. A project folder makes those pieces easy to run and review.
Colab is useful for GPU experiments, but this assignment has no GPU requirement.
Python 3.12 is the tested interpreter. Direct dependencies are pinned to the
versions used for verification; the pins are for reproducibility, not a claim
that they are the newest available versions.

## 2. Statistical rules instead of a trained model

The assignment explicitly specifies percent change, IQR/Z-score, and Pearson.
Twelve observations with no target labels do not justify a supervised model.
An Isolation Forest could produce an anomaly score, but introduces contamination,
feature scaling/interpretation concerns, and instability without establishing an
advantage over the requested transparent rules. An LLM is unnecessary because
sentences can be generated exactly from structured numeric evidence.

There is no learning objective to optimize and no validation set on which to
claim that a parameter is globally best. Defaults follow the brief. Useful
improvements here are correctness, explainability, data validation, and faithful
exports. A future learned detector would need more historical observations and
analyst-reviewed anomaly labels to evaluate added value.

## 3. IQR is the default; Z-score remains available

IQR uses the middle half of the data and does not require a normal distribution
to compute its fences. It is easy to explain from quartiles. It can still be
unstable on tiny or discrete samples, and it does not prove a data point is wrong.

The alternative Z-score uses the mean and population standard deviation of the
reference rows. Extreme observations can inflate the standard deviation and
reduce their own Z-score. In the supplied data, ANC=42 has pooled z about -2.955
and high-risk cases=28 has z about 2.712, so neither exceeds 3. This is an observed
method difference, not a reason to hide or change the calculation.

The default 1.5 multiplier is the assignment's conventional exploratory fence.
There is no claim that it optimizes recall or precision for healthcare anomalies.
Values exactly on a fence are not outliers, matching the strict inequalities.

## 4. Explicit reference populations

Pooled mode compares each indicator across all selected district-month rows,
matching the brief. It can mix time variation with district variation. Monthly
mode compares selected districts within each month. The UI exposes this choice
and each outlier records its scope and sample size.

Neither mode is a population-weighted state estimate or a leave-one-out model.
Each observed value is included in its reference distribution. Without exposure
denominators or a district sampling frame, the application uses descriptive
reference means and avoids calling them statewide coverage.

## 5. Prior month is retained when the view is filtered

A month filter describes which findings the user wants to see. If the user
chooses August, removing July before calculating changes would lose the expected
comparison. Trends therefore use full uploaded history for the chosen districts,
then filter their output periods. Only the previous calendar month is valid.

Outlier and correlation questions are about the selected cohort; those are
recalculated after filtering. The dashboard states these different semantics.
Line charts do not connect across calendar gaps.

## 6. Pearson correlation is exploratory

The required matrix uses pairwise complete data with at least three paired rows.
Two nonconstant points always produce absolute correlation one, so reporting
those as strong evidence would be misleading. Three is a computational guard,
not a sufficient sample size for reliable inference.

Each pair is emitted once. Its explanation reports paired rows, districts, and
months. Repeated rows are not independent; pooled relationships need not hold
within a district. No per-district correlation is claimed from its two points.
There are no p-values or invented confidence scores.

## 7. Severity is a declared policy

The brief requires data-derived Low/Medium/High severity but supplies no exact
formula. Every detection method therefore produces a score relative to its
active rule, and user-configurable multiples map that score to priority.

For a trend, a 30% change at a 10% threshold has score 3. Changing the threshold
to 20% reduces the score to 1.5. For an IQR outlier, a score of one corresponds to
the relevant fence; greater distance from the median increases the score.

The default 1.5 and 2.0 priority cutoffs are transparent engineering defaults,
not a medical standard. They should be agreed with stakeholders and evaluated
against reviewed findings if this becomes an operational system. Positive
outliers and improvements are still findings. The application intentionally
does not equate magnitude, bad performance, confidence, and clinical risk.

## 8. Missing values and schema errors are handled explicitly

Missing indicator values can be retained because computations can account for
their sample sizes. Invalid keys and duplicate records cannot be safely repaired
without additional information, so the loader rejects them. Counts are not
averaged to resolve duplicates; missing values are not silently filled.

The code accepts new district names, months, and numeric indicator columns
without changing narratives. Required healthcare columns remain enforced
because the assignment specifies them. Unit knowledge is isolated in config.

## 9. Small modules with conventional names

`data.py` handles loading, `analysis.py` calculates evidence, `narratives.py`
turns evidence into sentences, and `pipeline.py` coordinates a run. `app.py`
handles the UI. Names such as `frame`, `trends`, `previous`, `score`, `settings`,
and `result` describe their purpose without ornamental abstractions.

Dataclasses group a small number of related settings/results. There is no
plugin framework, separate API server, database, vector store, or model layer.
The same analysis functions serve both the dashboard and CLI, which reduces the
chance of their results diverging.

## 10. Verification matches the task

Tests check arithmetic, known findings, and failure modes that change the
meaning of the results. Renaming a district and editing a value must change the
sentences automatically. Shuffling input rows must not change exported findings.
Streamlit AppTest checks that filters, thresholds, empty states, and alternative
methods run without exceptions. JSON validity and CSV evidence are checked.

These checks establish implementation correctness for covered cases. They do
not establish clinical validity, population representativeness, predictive
accuracy, or production scalability. More real data and domain review are the
next requirements for those claims.

## 11. A labeled synthetic extension meets the size recommendation

The expanded default fixture has 12 districts and 12 consecutive months each,
including all 12 supplied rows. The untouched original file remains selectable.
This gives more useful UI/filter/history tests than merely duplicating rows.

The generator uses simple, declared baseline/noise/drift/count assumptions and
seed 42. It is not fitted to representative healthcare data. Three recorded
scenario changes test the detector, but their labels are not inputs to detection.
The seed is not searched to optimize correlations, and ordinary variation is
allowed to produce additional findings. See `data/README.md` for exact choices.

Provenance is separated from the six-column analysis CSV. A keyed sidecar labels
supplied and synthetic rows, and a checksum ties generation metadata to the CSV.
The dashboard and exported synthetic explanations are explicitly labeled.
Source changes reset data filters to avoid carrying an invalid selection from
one fixture into another. The original sample tests remain in the suite.

This meets the numerical dataset-size recommendation for a demonstration. It
does not imply real-world validity or that the employer has approved synthetic
data; the submission should disclose that choice plainly.
