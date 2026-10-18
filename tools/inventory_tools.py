"""
tools/inventory_tools.py
Deterministic Retail Inventory & Replenishment Math for StockSense AI.
Pure Python implementations of formulas:
- Lead Time Demand: D_L = sum_{t=1}^L d_hat_t
- Reorder Point: ROP = D_L + SS
- Reorder Quantity: Q = max(0, ceil(ROP - S - I))
- Order Cost: Cost = Q * Unit Cost
- Days of Cover: DOC = S / (D_L / L)
- Inventory Risk Classification (High, Medium, Low, Unknown)
"""

import math
from typing import List, Dict, Any, Tuple, Optional


def calculate_lead_time_demand(daily_forecasts: List[float], lead_time_days: int) -> float:
    """
    D_L = sum of forecast demand over the next L days.
    If forecasts list is shorter than lead_time_days, extrapolates using the average daily forecast.
    """
    if lead_time_days <= 0 or not daily_forecasts:
        return 0.0

    if len(daily_forecasts) >= lead_time_days:
        return round(float(sum(daily_forecasts[:lead_time_days])), 2)

    # If horizon is shorter than lead time, use average daily demand to scale
    avg_daily = float(sum(daily_forecasts)) / len(daily_forecasts)
    total = avg_daily * lead_time_days
    return round(total, 2)


def calculate_reorder_point(lead_time_demand: float, safety_stock: float) -> float:
    """
    ROP = D_L + SS
    Where D_L is lead time demand and SS is configured safety stock.
    """
    ss_clean = max(0.0, float(safety_stock)) if safety_stock is not None else 0.0
    dl_clean = max(0.0, float(lead_time_demand)) if lead_time_demand is not None else 0.0
    return round(dl_clean + ss_clean, 2)


def calculate_reorder_quantity(
    rop: Optional[float],
    current_stock: Optional[float],
    incoming_stock: Optional[float] = None
) -> Tuple[Optional[int], str]:
    """
    Q = max(0, ceil(ROP - S - I))
    Rules:
    - If current_stock (S) is None or missing, DO NOT calculate authoritative Q.
    - If incoming_stock (I) is None/unknown, DO NOT silently subtract it. Treat I = 0 with logged note.
    - Returns (Q, audit_explanation).
    """
    if current_stock is None or (isinstance(current_stock, float) and math.isnan(current_stock)):
        return None, "Authoritative reorder quantity unavailable: Current stock S is missing or unspecified."

    if rop is None or (isinstance(rop, float) and math.isnan(rop)):
        return None, "Authoritative reorder quantity unavailable: Reorder Point ROP cannot be determined."

    s = float(current_stock)
    if incoming_stock is not None and not math.isnan(incoming_stock):
        i = max(0.0, float(incoming_stock))
        incoming_note = f"confirmed incoming stock I={i:.0f} subtracted"
    else:
        i = 0.0
        incoming_note = "incoming stock I is unknown/unconfirmed (not subtracted)"

    net_deficit = rop - s - i
    if net_deficit <= 0:
        q = 0
        explanation = f"No reorder required (Q=0): S ({s:.0f}) + I ({i:.0f}) >= ROP ({rop:.1f})."
    else:
        q = int(math.ceil(net_deficit))
        explanation = f"Reorder Q={q} calculated as ceil(ROP {rop:.1f} - S {s:.0f} - I {i:.0f}) [{incoming_note}]."

    return q, explanation


def calculate_estimated_cost(
    quantity: Optional[int],
    unit_cost: Optional[float]
) -> Tuple[Optional[float], str]:
    """
    Cost = Q * Unit Cost
    Only calculate if unit_cost is available or explicitly configured.
    """
    if quantity is None:
        return None, "Cost unavailable: Reorder quantity is unspecified."

    if quantity == 0:
        return 0.0, "Cost is $0.00 since reorder quantity is 0."

    if unit_cost is None or (isinstance(unit_cost, float) and (math.isnan(unit_cost) or unit_cost <= 0)):
        return None, "Cost unavailable: Unit cost is missing or unspecified."

    cost = round(float(quantity) * float(unit_cost), 2)
    return cost, f"Estimated cost: {quantity} units * ${unit_cost:.2f}/unit = ${cost:.2f}."


def calculate_days_of_cover(
    current_stock: Optional[float],
    daily_demand: float
) -> Optional[float]:
    """
    Days of Cover (DOC) = S / daily_demand
    Returns None if stock is missing or daily_demand <= 0.
    """
    if current_stock is None or math.isnan(current_stock) or daily_demand <= 0:
        return None
    return round(float(current_stock) / float(daily_demand), 1)


def classify_inventory_risk(
    current_stock: Optional[float],
    lead_time_demand: Optional[float],
    rop: Optional[float]
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Classifies risk into High, Medium, Low, or Unknown.
    Rules:
    - High: S < D_L (Stock will exhaust before delivery arrives)
    - Medium: D_L <= S < ROP (Stock covers delivery window, but breaches reorder threshold)
    - Low: S >= ROP (Healthy inventory buffer)
    - Unknown: Missing S or D_L or invalid negative values
    """
    evidence = {
        "current_stock": current_stock,
        "lead_time_demand": lead_time_demand,
        "reorder_point": rop
    }

    if (
        current_stock is None or
        lead_time_demand is None or
        rop is None or
        (isinstance(current_stock, float) and math.isnan(current_stock)) or
        (isinstance(lead_time_demand, float) and math.isnan(lead_time_demand)) or
        (isinstance(rop, float) and math.isnan(rop)) or
        current_stock < 0
    ):
        return "Unknown", "Required inventory information is missing or invalid; risk level cannot be reliably asserted.", evidence

    s = float(current_stock)
    dl = float(lead_time_demand)
    r = float(rop)

    if s < dl:
        reason = f"CRITICAL STOCKOUT RISK: Available stock ({s:.0f}) is strictly less than lead-time demand ({dl:.1f}). Stockout imminent during replenishment lead time."
        return "High", reason, evidence
    elif dl <= s < r:
        reason = f"REORDER THRESHOLD BREACH: Stock ({s:.0f}) covers lead-time demand ({dl:.1f}) but is below Reorder Point ({r:.1f}). Breaches safety stock without replenishment."
        return "Medium", reason, evidence
    else:
        reason = f"HEALTHY BUFFER: Available stock ({s:.0f}) meets or exceeds Reorder Point ({r:.1f}). Buffer adequate."
        return "Low", reason, evidence
