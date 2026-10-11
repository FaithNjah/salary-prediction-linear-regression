"""
Train a simple linear regression model that predicts salary from years of experience.

Usage:
    python src/train.py
    python src/train.py --data data/salary_data.csv --model models/salary_model.joblib

The script:
    1. evaluates the model with 5-fold cross-validation (the dataset has only 30 rows,
       so a single train/test split would be unreliable);
    2. fits the final model on all of the data;
    3. saves it with joblib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

FEATURE = "YearsExperience"
TARGET = "Salary"


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df[[FEATURE]], df[TARGET]


def evaluate(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict[str, float]:
    """Cross-validated R², MAE and RMSE, computed over the out-of-fold
    predictions so that every row is predicted by a model that never saw it."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(LinearRegression(), X, y, cv=cv)
    return {
        "r2": r2_score(y, pred),
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def train(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    return LinearRegression().fit(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the salary prediction model.")
    parser.add_argument("--data", default="data/salary_data.csv")
    parser.add_argument("--model", default="models/salary_model.joblib")
    args = parser.parse_args()

    X, y = load_data(args.data)
    metrics = evaluate(X, y)
    print("5-fold cross-validation")
    print(f"  R²   {metrics['r2']:.3f}")
    print(f"  MAE  {metrics['mae']:,.0f}")
    print(f"  RMSE {metrics['rmse']:,.0f}")

    model = train(X, y)
    print("\nFinal model (trained on all data)")
    print(f"  Salary = {model.intercept_:,.2f} + {model.coef_[0]:,.2f} x YearsExperience")

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"\nSaved to {args.model}")


if __name__ == "__main__":
    main()
