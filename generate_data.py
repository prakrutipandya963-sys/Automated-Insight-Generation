"""Generate a synthetic demonstration panel while preserving the assignment sample."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from insights.config import PERCENT_COLUMNS, REQUIRED_COLUMNS
from insights.data import load_csv, validate_data


ROOT = Path(__file__).resolve().parent
EXTRA_DISTRICTS = ["Anand", "Bharuch", "Gandhinagar", "Jamnagar", "Junagadh", "Kheda"]


@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 42
    months: int = 12
    monthly_noise: float = 0.8
    persistence: float = 0.5
    annual_wave: float = 0.7
    monthly_drift_min: float = -0.1
    monthly_drift_max: float = 0.2


def build_dataset(sample_path=ROOT / "data" / "healthcare.csv", settings=None):
    settings = settings or GenerationSettings()
    if settings.months < 3:
        raise ValueError("Generate at least three months per district.")
    if settings.seed < 0 or not 0 <= settings.persistence < 1:
        raise ValueError("Use a non-negative seed and persistence in [0, 1).")
    if settings.monthly_noise < 0 or settings.annual_wave < 0:
        raise ValueError("Noise and annual-wave amplitudes must be non-negative.")
    if settings.monthly_drift_min > settings.monthly_drift_max:
        raise ValueError("The minimum monthly drift cannot exceed its maximum.")
    rng = np.random.default_rng(settings.seed)
    sample = load_csv(sample_path).frame
    periods = pd.date_range(end=sample.month.max(), periods=settings.months, freq="MS")
    if not set(sample.month).issubset(periods):
        raise ValueError("The generated date window must include every supplied sample month.")
    metrics = list(PERCENT_COLUMNS)
    districts = sorted(set(sample.district) | set(EXTRA_DISTRICTS))
    # Use the first supplied month as a scale reference, without learning the August shocks.
    first = sample.loc[sample.month.eq(sample.month.min())].set_index("district")
    median = first[metrics].median().to_numpy()
    source_keys = set(zip(sample.district, sample.month))
    anchor_index = list(periods).index(sample.month.min())
    rows, profiles = [], []
    for district in districts:
        if district in first.index:
            baseline = first.loc[district, metrics].to_numpy(dtype=float)
            case_mean = float(first.loc[district, "high_risk_cases"])
            origin = "First supplied month"
        else:
            # Shared district variation creates imperfect cross-indicator association.
            quality = rng.normal()
            baseline = median + quality * np.array([4.0, 3.5, 3.0]) + rng.normal(0, 0.8, 3)
            baseline = np.clip(baseline, 65, 98)
            case_mean = float(rng.uniform(first.high_risk_cases.min(), first.high_risk_cases.max()))
            origin = "Generated around sample baseline"
        drift = rng.uniform(settings.monthly_drift_min, settings.monthly_drift_max, 3)
        phase = rng.uniform(0, 2 * np.pi)
        noise = np.zeros(3)
        profiles.append({"district": district, "baseline_origin": origin,
                         "coverage_baseline": dict(zip(metrics, baseline.tolist())),
                         "monthly_drift": dict(zip(metrics, drift.tolist())),
                         "case_mean": case_mean, "wave_phase": float(phase)})
        for index, month in enumerate(periods):
            wave = settings.annual_wave * np.sin(2 * np.pi * index / 12 + phase)
            shared_noise = rng.normal(0, settings.monthly_noise * 0.5)
            noise = settings.persistence * noise + shared_noise + rng.normal(0, settings.monthly_noise, 3)
            coverage = baseline + drift * (index - anchor_index) + wave + noise
            coverage = np.round(np.clip(coverage, 0, 100), 1)
            # Counts are sampled independently of coverage; there is no invented causal formula.
            case_rate = case_mean * (1 + 0.05 * np.sin(2 * np.pi * index / 12 + phase))
            count = int(rng.poisson(case_rate))
            rows.append({"month": month, "district": district, **dict(zip(metrics, coverage)),
                         "high_risk_cases": count})
    frame = pd.DataFrame(rows).set_index(["district", "month"])

    # These are disclosed test scenarios, not claims about the named districts.
    events = []
    scenarios = [
        ("Bharuch", len(periods) // 3, "immunization", -18.0, "Temporary coverage dip"),
        ("Jamnagar", len(periods) // 2, "high_risk_cases", 22, "Temporary case-count spike"),
        ("Kheda", len(periods) // 2 + 1, "anc_coverage", -16.0, "Temporary ANC reporting dip"),
    ]
    for district, index, indicator, delta, description in scenarios:
        month = periods[min(index, len(periods) - 1)]
        if (district, month) in source_keys:
            raise ValueError("A generated scenario must not modify a supplied row.")
        previous = float(frame.loc[(district, month), indicator])
        updated = previous + delta
        if indicator in metrics:
            updated = round(float(np.clip(updated, 0, 100)), 1)
        else:
            updated = int(max(updated, 0))
        frame.loc[(district, month), indicator] = updated
        events.append({"district": district, "month": f"{month:%Y-%m}", "indicator": indicator,
                       "description": description, "generated_baseline": previous,
                       "final_value": updated, "change": updated - previous})

    # Replace entire keyed rows, ensuring all four values match the supplied sample exactly.
    source = sample.set_index(["district", "month"])
    frame.loc[source.index, list(source.columns)] = source
    frame = frame.reset_index()[list(REQUIRED_COLUMNS)].sort_values(["district", "month"]).reset_index(drop=True)
    validated = validate_data(frame)
    frame = validated.frame[list(REQUIRED_COLUMNS)]
    frame["high_risk_cases"] = frame["high_risk_cases"].astype(int)
    provenance = frame[["month", "district"]].copy()
    provenance["source"] = ["assignment_sample" if (row.district, row.month) in source_keys else "synthetic"
                            for row in frame.itertuples(index=False)]
    metadata = {
        "name": "Synthetic demonstration with preserved assignment sample",
        "data_type": "synthetic_with_assignment_sample",
        "disclaimer": "Demonstration data, not observed healthcare statistics. Supplied sample rows are illustrative and not independently verified real observations.",
        "rows": len(frame), "districts": len(districts), "months_per_district": settings.months,
        "start_month": periods[0].strftime("%Y-%m"), "end_month": periods[-1].strftime("%Y-%m"),
        "assignment_rows": len(sample), "generated_rows": len(frame) - len(sample),
        "source": "Assignment 4 - Automated Insight Generation, PDF section 2.1",
        "source_csv_sha256": sha256(Path(sample_path).read_bytes()).hexdigest(),
        "generation_settings": asdict(settings), "district_profiles": profiles,
        "scenarios": events,
        "units": {**{column: "percent, 0 to 100" for column in metrics}, "high_risk_cases": "whole count"},
        "scope_note": "Small illustrative reporting counts matching the supplied sample scale, not a calibrated model of district population totals.",
        "notes": [
            "The seed fixes a reproducible draw; it was not selected by searching for stronger correlations.",
            "The annual wave is a simulation device, not an estimated seasonal effect.",
            "Generated coverage correlations partly reflect the shared district baseline and noise by construction.",
            "No coverage denominator, patient population, or clinical target is inferred.",
            "Scenario changes are recorded only in metadata, never passed as labels to the detector.",
            "Supplied rows are preserved exactly; their boundary with generated history can produce additional changes.",
            "Meeting size recommendations does not establish real-world statistical validity or evaluator acceptance of synthetic data.",
        ],
    }
    return frame, provenance, metadata


def save_dataset(frame, provenance, metadata, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, date_format="%Y-%m", float_format="%.1f", lineterminator="\n")
    path = folder / "healthcare_synthetic.csv"
    path.write_text(content, encoding="utf-8")
    metadata = {**metadata, "csv_sha256": sha256(content.encode()).hexdigest()}
    (folder / "healthcare_synthetic.metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    provenance.to_csv(folder / "row_provenance.csv", index=False, date_format="%Y-%m", lineterminator="\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    frame, provenance, metadata = build_dataset(settings=GenerationSettings(seed=args.seed, months=args.months))
    path = save_dataset(frame, provenance, metadata, args.output)
    print(f"Saved {len(frame)} rows to {path}")
    print(f"{metadata['districts']} districts x {metadata['months_per_district']} months; "
          f"{metadata['assignment_rows']} supplied rows + {metadata['generated_rows']} generated rows.")
    print(metadata["disclaimer"])


if __name__ == "__main__":
    main()
