"""
core/inventory_agent.py
Agent 3: Inventory Risk Analysis Agent
Combines demand forecasts with inventory stock positions.
Calculates Lead Time Demand (D_L), Reorder Point (ROP), and Days of Cover (DOC).
Classifies products into High, Medium, Low, or Unknown stockout risk.
"""

from typing import Dict, List, Any, Optional
import pandas as pd

from core.forecaster import ForecastResult
from tools.inventory_tools import (
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_days_of_cover,
    classify_inventory_risk
)


class RiskAssessment:
    def __init__(
        self,
        product_id: str,
        store_id: str,
        current_stock: Optional[float],
        lead_time_days: int,
        safety_stock: float,
        lead_time_demand: float,
        reorder_point: float,
        days_of_cover: Optional[float],
        risk_level: str,
        risk_reason: str,
        evidence: Dict[str, Any]
    ):
        self.product_id = product_id
        self.store_id = store_id
        self.current_stock = current_stock
        self.lead_time_days = lead_time_days
        self.safety_stock = safety_stock
        self.lead_time_demand = lead_time_demand
        self.reorder_point = reorder_point
        self.days_of_cover = days_of_cover
        self.risk_level = risk_level
        self.risk_reason = risk_reason
        self.evidence = evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "store_id": self.store_id,
            "current_stock": self.current_stock,
            "lead_time_days": self.lead_time_days,
            "safety_stock": self.safety_stock,
            "lead_time_demand": self.lead_time_demand,
            "reorder_point": self.reorder_point,
            "days_of_cover": self.days_of_cover,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "evidence": self.evidence
        }


class InventoryRiskAgent:
    """
    Agent 3: Deterministic Inventory Risk Analyst.
    """

    def __init__(self, lead_time_days: int = 7, safety_stock: float = 14.0):
        self.lead_time_days = lead_time_days
        self.safety_stock = safety_stock

    def assess_risk(
        self,
        forecast_res: ForecastResult,
        current_stock: Optional[float]
    ) -> RiskAssessment:
        """
        Assesses stockout risk for a given product forecast and current stock level.
        """
        # Calculate lead-time demand D_L
        d_l = calculate_lead_time_demand(
            daily_forecasts=forecast_res.forecast_sales,
            lead_time_days=self.lead_time_days
        )

        # Calculate Reorder Point ROP
        rop = calculate_reorder_point(lead_time_demand=d_l, safety_stock=self.safety_stock)

        # Calculate Days of Cover DOC
        doc = calculate_days_of_cover(
            current_stock=current_stock,
            daily_demand=forecast_res.average_daily_demand
        )

        # Classify risk
        risk_level, reason, evidence = classify_inventory_risk(
            current_stock=current_stock,
            lead_time_demand=d_l,
            rop=rop
        )

        evidence.update({
            "lead_time_days": self.lead_time_days,
            "safety_stock": self.safety_stock,
            "average_daily_demand": forecast_res.average_daily_demand,
            "days_of_cover": doc
        })

        return RiskAssessment(
            product_id=forecast_res.product_id,
            store_id=forecast_res.store_id,
            current_stock=current_stock,
            lead_time_days=self.lead_time_days,
            safety_stock=self.safety_stock,
            lead_time_demand=d_l,
            reorder_point=rop,
            days_of_cover=doc,
            risk_level=risk_level,
            risk_reason=reason,
            evidence=evidence
        )

    def run_all(
        self,
        forecast_results: Dict[str, ForecastResult],
        stock_lookup: Dict[str, Optional[float]]
    ) -> Dict[str, RiskAssessment]:
        """
        Runs risk assessment across all product forecasts.
        """
        assessments: Dict[str, RiskAssessment] = {}
        for key, fc in forecast_results.items():
            stock = stock_lookup.get(key)
            if stock is None:
                # Also check by product_id alone if key has store suffix
                stock = stock_lookup.get(fc.product_id)
            assessments[key] = self.assess_risk(fc, current_stock=stock)
        return assessments
