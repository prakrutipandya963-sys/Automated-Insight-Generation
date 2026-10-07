import argparse
from pathlib import Path

from insights.config import Settings, Target
from insights.data import DataError, load_csv, print_report
from insights.pipeline import analyze, save_results


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible district healthcare insights.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "healthcare_synthetic.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--trend-threshold", type=float, default=10.0)
    parser.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr")
    parser.add_argument("--outlier-scope", choices=["pooled", "monthly"], default="pooled")
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--correlation-threshold", type=float, default=0.7)
    parser.add_argument("--min-outlier-rows", type=int, default=4)
    parser.add_argument("--min-correlation-rows", type=int, default=3)
    parser.add_argument("--medium-ratio", type=float, default=1.5)
    parser.add_argument("--high-ratio", type=float, default=2.0)
    parser.add_argument("--districts", nargs="+")
    parser.add_argument("--months", nargs="+", help="Months in YYYY-MM format")
    parser.add_argument("--indicators", nargs="+")
    parser.add_argument("--target-indicator")
    parser.add_argument("--target-value", type=float)
    parser.add_argument("--target-direction", choices=["minimum", "maximum"], default="minimum")
    args = parser.parse_args()
    if (args.target_indicator is None) != (args.target_value is None):
        parser.error("Provide both --target-indicator and --target-value.")
    try:
        settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
        dataset = load_csv(args.input)
        print_report(dataset)
        target = Target(args.target_indicator, args.target_value, args.target_direction) if args.target_indicator else None
        result = analyze(dataset, settings, districts=args.districts, months=args.months,
                         indicators=args.indicators, target=target)
        save_results(result, args.output)
    except (DataError, ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"\nSaved {len(result.insights)} insights to {args.output.resolve()}")
    print(result.insights[["insight_id", "type", "indicator", "entity", "severity"]].to_string(index=False))
    for warning in result.warnings:
        print(f"Note: {warning}")


if __name__ == "__main__":
    main()
