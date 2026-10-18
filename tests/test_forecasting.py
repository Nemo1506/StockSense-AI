"""
tests/test_forecasting.py
Automated tests for time-series demand forecasting:
- Strictly chronological splitting
- 7-Day Moving Average calculation
- Adaptive window handling for short history
- MAE, RMSE, and Zero-Safe WAPE calculations
- Zero-sales and edge-case protection
"""

import pandas as pd
import numpy as np
import pytest

from tools.forecast_tools import (
    split_chronological,
    forecast_7day_moving_average,
    forecast_secondary_ridge,
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    evaluate_forecast_on_holdout
)


def test_chronological_forecast_splitting():
    dates = pd.date_range("2026-08-01", periods=14, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": [10 + i for i in range(14)]
    })

    train_df, test_df = split_chronological(df, date_col="date", test_days=4)
    assert len(train_df) == 10
    assert len(test_df) == 4
    # Ensure test dates are strictly after train dates (no data leakage)
    assert train_df["date"].max() < test_df["date"].min()


def test_7day_moving_average_math():
    # Exactly 7 days: [10, 10, 10, 10, 10, 10, 10] -> mean is 10.0
    history = [10.0] * 7
    preds = forecast_7day_moving_average(history, horizon_days=3)
    assert len(preds) == 3
    assert preds[0] == 10.0
    assert preds[1] == 10.0
    assert preds[2] == 10.0

    # Unequal values: [7, 8, 9, 10, 11, 12, 13] -> sum=70, mean=10.0
    history_inc = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    preds_inc = forecast_7day_moving_average(history_inc, horizon_days=1)
    assert preds_inc[0] == 10.0


def test_adaptive_moving_average_short_history():
    # Only 3 days of history
    history = [2.0, 4.0, 6.0]
    preds = forecast_7day_moving_average(history, horizon_days=2)
    # Mean of 3 days is 4.0
    assert preds[0] == 4.0
    assert preds[1] == 4.0


def test_forecast_empty_history():
    preds = forecast_7day_moving_average([], horizon_days=5)
    assert preds == [0.0] * 5


def test_forecast_secondary_ridge():
    dates = pd.date_range("2026-08-01", periods=20, freq="D")
    sales = [float(5 + (i % 7) * 2) for i in range(20)]
    preds = forecast_secondary_ridge(list(dates), sales, horizon_days=7)
    assert len(preds) == 7
    assert all(p >= 0.0 for p in preds)


def test_mae_and_rmse():
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 35.0]
    # Errors: |10-12|=2, |20-18|=2, |30-35|=5 -> mean = 9/3 = 3.0
    assert calculate_mae(actual, predicted) == 3.0
    # Squared errors: 4, 4, 25 -> sum = 33 / 3 = 11 -> sqrt(11) = 3.317
    assert calculate_rmse(actual, predicted) == pytest.approx(3.317, 0.01)


def test_zero_sales_wape_safety():
    # If all actual sales are zero, WAPE denominator is 0. Must return None, not raise ZeroDivisionError!
    actual_zero = [0.0, 0.0, 0.0]
    predicted = [2.0, 1.0, 0.5]
    wape = calculate_wape(actual_zero, predicted)
    assert wape is None

    # Normal case: actual sales > 0
    actual_normal = [10.0, 20.0, 10.0]  # sum = 40
    predicted_normal = [12.0, 18.0, 10.0]  # err sum = 2 + 2 + 0 = 4 -> 4/40 = 0.10
    wape_normal = calculate_wape(actual_normal, predicted_normal)
    assert wape_normal == 0.10


def test_evaluate_forecast_on_holdout():
    train = [10.0] * 14
    holdout = [10.0, 10.0, 10.0]
    eval_res = evaluate_forecast_on_holdout(train, holdout, model_name="7-SMA")
    assert eval_res["mae"] == 0.0
    assert eval_res["rmse"] == 0.0
    assert eval_res["wape"] == 0.0
