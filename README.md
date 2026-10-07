# Synthetic healthcare dataset

This dataset was created for testing Assignment 4: Automated Insight Generation.
It is **not an official HMIS extract, a survey, or observed healthcare data**.
District names are recognizable labels and do not imply real performance.

## What is included

| Property | Value |
| --- | --- |
| Main file | `healthcare_synthetic.csv` |
| Districts | 12 |
| Months per district | 12 consecutive months |
| Period | 2025-09 through 2026-08 |
| Total rows | 144 |
| Supplied rows preserved | 12 |
| Newly generated rows | 132 |
| Missing indicator values | None |
| Duplicate district-month keys | None |
| Random seed | 42, fixed for reproducibility |

District labels: Ahmedabad, Anand, Bharuch, Bhavnagar, Gandhinagar, Jamnagar,
Junagadh, Kheda, Mehsana, Rajkot, Surat, and Vadodara.

The provided sample requires no external dataset link. It is kept unchanged as
`healthcare.csv`. Every July/August value supplied for its six districts is copied
by district-month key into the expanded dataset. The source is the assignment
PDF, section 2.1. Those supplied rows are illustrative; their status as actual
observations has not been independently verified.

## Data dictionary

| Column | Type and unit | Meaning in this fixture |
| --- | --- | --- |
| `month` | `YYYY-MM` | Reporting month, one row per district/month |
| `district` | Text | District label |
| `anc_coverage` | Numeric, 0-100 percent | Supplied or simulated ANC coverage indicator |
| `institutional_delivery` | Numeric, 0-100 percent | Supplied or simulated institutional delivery indicator |
| `immunization` | Numeric, 0-100 percent | Supplied or simulated immunization coverage indicator |
| `high_risk_cases` | Non-negative integer count | Supplied or simulated monthly reporting count |

Percentages use the assignment's convention: `85` means 85%, not 0.85. No patient
records, denominator counts, prevalence estimates, or clinical thresholds are
invented. The PDF does not define ANC visit eligibility, immunization schedule,
or coverage denominators, so the generator does not pretend to supply them.

## How generation works

`generate_data.py` contains the complete generation logic and can be rerun using
the project's Python dependencies. It does not call an LLM or external API.

1. Read and validate the original assignment CSV.
2. End the generated window at the latest supplied month, August 2026. Include
   12 months, giving September 2025 through August 2026. These dates are a test
   window, not a claim that historical measurements were collected.
3. For the six supplied districts, use the first supplied month's values as
   baseline scales. Avoid using the August drops as ordinary baselines.
4. For six additional district labels, draw coverage baselines around the first
   sample month's median, with shared district variation and indicator-specific
   noise. This creates some association between coverage indicators without
   making them identical. Baselines are constrained to 65-98 percent as a
   simulation range, not a clinical or empirical reference range.
5. Add mild per-indicator drift, a small annual wave, and persistent monthly
   variation. Round coverage to one decimal and bound it to 0-100.
6. Draw counts from a Poisson distribution around the supplied first-month case
   scale. New district means are drawn between the sample's minimum and maximum.
   A small annual wave changes the rate. Counts are not calculated from ANC
   percentages, so no causal ANC-to-risk relationship is imposed.
7. Apply three disclosed test scenarios to generated rows only.
8. Copy all supplied rows over their matching generated rows, then validate and
   export the final panel. This preserves all original values exactly.

## Parameters and their justification

| Parameter | Default | Purpose and limitation |
| --- | --- | --- |
| Seed | 42 | Repeat the same draw; no statistical meaning and no seed search |
| Districts / months | 12 / 12 | Exceed 10 districts and 3 months, with a manageable year-long demo |
| Monthly coverage innovation SD | 0.8 percentage points | Modest local variability; an engineering assumption, not an estimate |
| Noise persistence | 0.5 | Some continuity from one month to the next; not fitted |
| Shared monthly innovation SD | 0.4 percentage points | Allows coverage indicators to move together imperfectly |
| Annual-wave amplitude | 0.7 percentage points | Small simulated periodic movement; not measured seasonality |
| Monthly coverage drift | Uniform from -0.1 to +0.2 percentage points | Permits modest decline or improvement without deterministic perfection |
| New-district common baseline scales | 4.0 / 3.5 / 3.0 percentage points | Variation in ANC / delivery / immunization around the sample median |
| New-district independent baseline SD | 0.8 percentage points | Avoids perfectly linked district profiles |
| Case-count annual rate variation | 5% | Small bounded change in the count-generating rate |
| Coverage limits | 0-100 | Mathematical bounds of percentages |

These are explicit simulation choices, not learned best parameters or validated
healthcare distributions. Realism here means bounded values, integer counts,
district heterogeneity, temporal continuity, and imperfect relationships at the
assignment's illustrative scale. It does not mean matching real district
populations, incidence rates, health programs, or local seasonality.

## Disclosed test scenarios

| District label | Month | Indicator | Added change to generated baseline |
| --- | --- | --- | --- |
| Bharuch | 2026-01 | Immunization | -18 percentage points |
| Jamnagar | 2026-03 | High-risk cases | +22 cases |
| Kheda | 2026-04 | ANC coverage | -16 percentage points |

For seed 42, their final values are 71.8%, 36 cases, and 67.1%, respectively.
Descriptions such as a reporting dip are fictional test scenarios, not asserted
causes. Magnitudes create interpretable changes; they are not clinical estimates.
The following month returns to its ordinary generated value, so recovery may
also trigger a trend finding. The supplied Ahmedabad/Mehsana changes remain.

The analytics code does not import the generator or use scenario labels.
Metadata records changes for audit and test verification only. Additional
outliers and percentage-change flags can arise naturally. The scenario list is
not an exhaustive ground-truth anomaly label set, so precision/F1 claims are not
made from it.

## Expected numerical checks

| Indicator | Minimum | Maximum |
| --- | ---: | ---: |
| ANC coverage | 42.0 | 95.0 |
| Institutional delivery | 79.7 | 97.6 |
| Immunization | 71.8 | 98.3 |
| High-risk cases | 3 | 36 |

The default rules find 113 monthly-change insights, 30 IQR outlier insights, and
one correlation insight. These are findings, not distinct affected rows.
Small count changes frequently exceed 10%; that limitation is intentionally
visible. Pooled IQR can repeatedly flag a district with a different stable
baseline. The generated delivery/immunization correlation is partly a result
of the shared simulation structure and must not be presented as new evidence.

## Provenance and reproducibility

- `row_provenance.csv` has the keys `month` and `district` plus `source`, whose
  values are `assignment_sample` or `synthetic`. It is an audit sidecar, not an
  analytics input. The main CSV retains exactly the six required columns.
- `healthcare_synthetic.metadata.json` records the seed, settings, district
  profiles, test events, disclaimers, and source/final CSV SHA-256 checksums.
- Loading the CSV by path automatically attaches its matching metadata sidecar.
  The dashboard also recognizes an exact uploaded copy of the bundled CSV.
- A changed CSV with mismatched metadata is labeled unverified. Re-exported or
  renamed copies may lose automatic recognition; keep the dataset labeled
  synthetic and preserve its metadata when sharing it.
- Every insight carries `data_origin` and a "Synthetic demonstration:" explanation
  prefix when matched synthetic metadata is loaded. The full report additionally
  includes generation provenance in `run_metadata.json`. Share the full report
  for the complete audit details.

Reproduce the bundled fixture from the project folder:

```bash
python generate_data.py --seed 42 --months 12
python run_analysis.py
python -m pytest -q
```

For a separate minimum-size demonstration without overwriting the bundled file:

```bash
python generate_data.py --months 3 --output data/demo_3_months
python run_analysis.py --input data/demo_3_months/healthcare_synthetic.csv --output outputs/demo_3_months
```

The expanded fixture satisfies the requested schema and numerical size
recommendations. It does not increase the number of independently observed real
districts or prove clinical/statistical validity. The submission should state
plainly that it is a synthetic extension of the provided illustrative sample.
