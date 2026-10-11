"""
Predict salary from years of experience with the trained model.

Usage:
    python src/predict.py 5
    python src/predict.py 1.5 3 7.5 12
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict salary from years of experience.")
    parser.add_argument("years", type=float, nargs="+", help="years of experience")
    parser.add_argument("--model", default="models/salary_model.joblib")
    args = parser.parse_args()

    if any(y < 0 for y in args.years):
        parser.error("years of experience cannot be negative")

    model = joblib.load(args.model)
    X = pd.DataFrame({"YearsExperience": args.years})
    for years, salary in zip(args.years, model.predict(X)):
        note = "  (outside the 1.2–10.6 years seen in training)" if years < 1.2 or years > 10.6 else ""
        print(f"{years:>5.1f} years -> {salary:>10,.0f}{note}")


if __name__ == "__main__":
    main()
