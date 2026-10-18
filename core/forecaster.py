"""
core/forecaster.py
Agent 2: Demand Forecasting Agent
Aggregates sales by date and product/store.
Executes chronological train/validation splits without data leakage.
Generates 7-Day Moving Average baseline and secondary Ridge lag forecasts.
Computes evaluation metrics (MAE, RMSE, and Zero-Safe WAPE).
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from tools.forecast_tools import (
    split_chronological,
    forecast_7day_moving_average,
    forecast_secondary_ridge,
    evaluate_forecast_on_holdout
)


class ForecastResult:
    def __init__(
        self,
        product_id: str,
        store_id: str,
        historical_dates: List[str],
        historical_sales: List[float],
        forecast_dates: List[str],
        forecast_sales: List[float],
        model_name: str,
        metrics: Dict[str, Any],
        average_daily_demand: float
    ):
        self.product_id = product_id
        self.store_id = store_id
        self.historical_dates = historical_dates
        self.historical_sales = historical_sales
        self.forecast_dates = forecast_dates
        self.forecast_sales = forecast_sales
        self.model_name = model_name
        self.metrics = metrics
        self.average_daily_demand = average_daily_demand

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "store_id": self.store_id,
            "historical_dates": self.historical_dates,
            "historical_sales": self.historical_sales,
            "forecast_dates": self.forecast_dates,
            "forecast_sales": self.forecast_sales,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "average_daily_demand": self.average_daily_demand
        }


class DemandForecastingAgent:
    """
    Agent 2: Conducts transparent, chronological time-series forecasting.
    """

    def __init__(self, forecast_model_choice: str = "7-SMA", horizon_days: int = 7):
        self.forecast_model_choice = forecast_model_choice
        self.horizon_days = horizon_days

    def forecast_product(
        self,
        df_product: pd.DataFrame,
        product_id: str,
        store_id: str = "ALL_STORES"
    ) -> ForecastResult:
        """
        Executes forecasting for a single (product, store) time series.
        """
        # Aggregate daily sales
        daily = df_product.groupby("_parsed_date")["_parsed_sales"].sum().reset_index()
        daily = daily.sort_values("_parsed_date").reset_index(drop=True)

        # Build contiguous date index if gaps exist
        if not daily.empty:
            full_idx = pd.date_range(
                start=daily["_parsed_date"].min(),
                end=daily["_parsed_date"].max(),
                freq="D"
            )
            daily = daily.set_index("_parsed_date").reindex(full_idx, fill_value=0.0).reset_index()
            daily.rename(columns={"index": "_parsed_date"}, inplace=True)

        dates_list = daily["_parsed_date"].dt.strftime("%Y-%m-%d").tolist()
        sales_list = daily["_parsed_sales"].tolist()

        # Holdout validation (last 7 days or 25% if shorter)
        holdout_days = min(7, max(1, len(sales_list) // 4)) if len(sales_list) > 3 else 0
        if holdout_days > 0 and len(sales_list) > holdout_days:
            train_sales = sales_list[:-holdout_days]
            test_sales = sales_list[-holdout_days:]
            eval_metrics = evaluate_forecast_on_holdout(
                historical_sales=train_sales,
                holdout_sales=test_sales,
                model_name=self.forecast_model_choice
            )
        else:
            eval_metrics = {
                "model": self.forecast_model_choice,
                "mae": 0.0,
                "rmse": 0.0,
                "wape": None,
                "wape_str": "N/A (Insufficient history for holdout)",
                "holdout_samples": 0
            }

        # Multi-step forward forecast using all available historical observations
        last_date = daily["_parsed_date"].iloc[-1] if not daily.empty else pd.Timestamp.now()
        future_dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(self.horizon_days)
        ]

        if self.forecast_model_choice == "Ridge Regression" and len(sales_list) >= 8:
            future_preds = forecast_secondary_ridge(
                dates=daily["_parsed_date"].tolist(),
                sales=sales_list,
                horizon_days=self.horizon_days
            )
        else:
            future_preds = forecast_7day_moving_average(
                history_sales=sales_list,
                horizon_days=self.horizon_days
            )

        avg_demand = round(float(np.mean(future_preds)) if future_preds else 0.0, 2)

        return ForecastResult(
            product_id=product_id,
            store_id=store_id,
            historical_dates=dates_list,
            historical_sales=sales_list,
            forecast_dates=future_dates,
            forecast_sales=future_preds,
            model_name=self.forecast_model_choice,
            metrics=eval_metrics,
            average_daily_demand=avg_demand
        )

    def run_all(
        self,
        clean_df: pd.DataFrame,
        products: Optional[List[str]] = None
    ) -> Dict[str, ForecastResult]:
        """
        Runs forecasting across all target products in the cleaned dataset.
        Returns a dictionary keyed by product_id (or product_id::store_id).
        """
        results: Dict[str, ForecastResult] = {}
        target_prods = products if products else clean_df["_product_id"].unique().tolist()

        for pid in target_prods:
            sub_df = clean_df[clean_df["_product_id"] == pid]
            if sub_df.empty:
                continue

            # Primary store breakdown (or default)
            store_ids = sub_df["_store_id"].unique().tolist()
            for sid in store_ids:
                store_df = sub_df[sub_df["_store_id"] == sid]
                key = f"{pid}::{sid}" if len(store_ids) > 1 else pid
                res = self.forecast_product(store_df, product_id=pid, store_id=sid)
                results[key] = res

        return results
