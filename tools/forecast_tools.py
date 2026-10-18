"""
tools/forecast_tools.py
Deterministic Time-Series Forecasting Tools for StockSense AI.
Includes:
- Chronological split (strictly non-random)
- 7-Day Simple Moving Average (7-SMA) baseline
- Ridge / Linear Lag-Regression secondary model
- MAE, RMSE, and Zero-Safe WAPE calculations
- Multi-step forward demand projection
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def split_chronological(
    df: pd.DataFrame,
    date_col: str,
    test_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically into train and validation sets.
    The most recent `test_days` form the test set; older records form the train set.
    NEVER shuffles or randomly samples.
    """
    if df.empty:
        return df.copy(), df.copy()

    df_sorted = df.sort_values(by=date_col).copy()
    unique_dates = df_sorted[date_col].drop_duplicates().sort_values().tolist()

    if len(unique_dates) <= test_days:
        # If dataset has fewer or equal unique dates than test_days,
        # use at least 1 day for test and remainder for train, or 50/50
        split_idx = max(1, len(unique_dates) - max(1, len(unique_dates) // 3))
    else:
        split_idx = len(unique_dates) - test_days

    split_date = unique_dates[split_idx]
    train_df = df_sorted[df_sorted[date_col] < split_date].copy()
    test_df = df_sorted[df_sorted[date_col] >= split_date].copy()

    return train_df, test_df


def forecast_7day_moving_average(
    history_sales: List[float],
    horizon_days: int = 7
) -> List[float]:
    """
    Forecasts demand for the next `horizon_days` using a 7-day Simple Moving Average (7-SMA).
    Formula:
        d_hat_{t+1} = (1/7) * sum_{i=0}^6 d_{t-i}
    Adapts if historical observations < 7 (uses mean of all available history).
    If history is completely empty, returns zeroes.
    Multi-step projection autoregressively rolls forecasted values forward.
    """
    if not history_sales:
        return [0.0] * horizon_days

    clean_history = [float(x) if (x is not None and not math.isnan(x)) else 0.0 for x in history_sales]
    if not clean_history:
        return [0.0] * horizon_days

    forecasts = []
    current_window = list(clean_history)

    for _ in range(horizon_days):
        k = min(7, len(current_window))
        if k == 0:
            pred = 0.0
        else:
            recent_k = current_window[-k:]
            pred = float(np.mean(recent_k))
        # Ensure non-negative forecast
        pred = max(0.0, round(pred, 2))
        forecasts.append(pred)
        current_window.append(pred)

    return forecasts


def forecast_secondary_ridge(
    dates: List[pd.Timestamp],
    sales: List[float],
    horizon_days: int = 7
) -> List[float]:
    """
    Secondary forecasting model using Ridge regression with lag and calendar features.
    Features: Lag-1, Lag-7, 7-day rolling mean, day of week.
    Falls back cleanly to moving average if series has insufficient history (< 8 points).
    """
    if len(sales) < 8:
        return forecast_7day_moving_average(sales, horizon_days=horizon_days)

    df = pd.DataFrame({"date": dates, "sales": sales}).sort_values("date").reset_index(drop=True)
    df["sales"] = df["sales"].fillna(0.0).clip(lower=0.0)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_7"] = df["sales"].shift(1).rolling(window=7, min_periods=1).mean()
    df["dayofweek"] = df["date"].dt.dayofweek

    # Drop NaNs created by lag_7
    train_data = df.dropna().copy()
    if len(train_data) < 5:
        return forecast_7day_moving_average(sales, horizon_days=horizon_days)

    feature_cols = ["lag_1", "lag_7", "rolling_7", "dayofweek"]
    X_train = train_data[feature_cols].values
    y_train = train_data["sales"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Iterative forward forecast
    last_date = df["date"].iloc[-1]
    extended_sales = list(df["sales"].values)
    forecasts = []

    for step in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=step)
        lag_1 = extended_sales[-1]
        lag_7 = extended_sales[-7] if len(extended_sales) >= 7 else np.mean(extended_sales)
        rolling_7 = np.mean(extended_sales[-7:]) if len(extended_sales) >= 7 else np.mean(extended_sales)
        dow = future_date.dayofweek

        x_feat = np.array([[lag_1, lag_7, rolling_7, dow]])
        pred = float(model.predict(x_feat)[0])
        pred = max(0.0, round(pred, 2))
        forecasts.append(pred)
        extended_sales.append(pred)

    return forecasts


def calculate_mae(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Error (MAE): (1/n) * sum |y - y_hat|"""
    if not actual or not predicted or len(actual) != len(predicted):
        return 0.0
    errors = [abs(float(a) - float(p)) for a, p in zip(actual, predicted)]
    return round(float(np.mean(errors)), 3)


def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error (RMSE): sqrt( (1/n) * sum (y - y_hat)^2 )"""
    if not actual or not predicted or len(actual) != len(predicted):
        return 0.0
    squared_errors = [(float(a) - float(p)) ** 2 for a, p in zip(actual, predicted)]
    return round(float(math.sqrt(np.mean(squared_errors))), 3)


def calculate_wape(actual: List[float], predicted: List[float]) -> Optional[float]:
    """
    Weighted Absolute Percentage Error (WAPE): sum |y - y_hat| / sum y.
    Strictly handles zero-sales cases: returns None if sum(actual) == 0 to prevent division by zero.
    """
    if not actual or not predicted or len(actual) != len(predicted):
        return None
    sum_actual = float(np.sum(actual))
    if sum_actual <= 0.0:
        return None  # Zero actual demand; WAPE is mathematically undefined

    sum_abs_err = float(np.sum([abs(float(a) - float(p)) for a, p in zip(actual, predicted)]))
    return round(sum_abs_err / sum_actual, 4)


def evaluate_forecast_on_holdout(
    historical_sales: List[float],
    holdout_sales: List[float],
    model_name: str = "7-SMA"
) -> Dict[str, Any]:
    """
    Evaluates forecasting model against a chronological holdout validation set.
    """
    k_holdout = len(holdout_sales)
    if k_holdout == 0:
        return {
            "model": model_name,
            "mae": 0.0,
            "rmse": 0.0,
            "wape": None,
            "wape_str": "N/A (Empty holdout)",
            "holdout_samples": 0
        }

    preds = forecast_7day_moving_average(historical_sales, horizon_days=k_holdout)

    mae = calculate_mae(holdout_sales, preds)
    rmse = calculate_rmse(holdout_sales, preds)
    wape = calculate_wape(holdout_sales, preds)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "wape_str": f"{wape * 100:.2f}%" if wape is not None else "N/A (Total sales = 0)",
        "holdout_samples": k_holdout,
        "predictions": preds,
        "actuals": holdout_sales
    }
