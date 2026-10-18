"""
tests/test_replenishment.py
Automated tests for deterministic inventory arithmetic, risk classification,
and replenishment quantities:
- Lead Time Demand D_L
- Reorder Point ROP = D_L + SS
- Reorder Quantity Q = max(0, ceil(ROP - S - I))
- Cost = Q * Unit Cost
- Missing stock and incoming stock handling
- Risk stratification (High, Medium, Low, Unknown)
"""

import pytest

from tools.inventory_tools import (
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_reorder_quantity,
    calculate_estimated_cost,
    calculate_days_of_cover,
    classify_inventory_risk
)


def test_lead_time_demand_exact():
    forecasts = [10.0, 12.0, 14.0, 10.0, 8.0, 11.0, 15.0]
    # L = 3 days -> 10 + 12 + 14 = 36.0
    dl_3 = calculate_lead_time_demand(forecasts, lead_time_days=3)
    assert dl_3 == 36.0

    # L = 7 days -> sum = 80.0
    dl_7 = calculate_lead_time_demand(forecasts, lead_time_days=7)
    assert dl_7 == 80.0


def test_lead_time_demand_extrapolation():
    # If forecasts has 3 days [10, 10, 10] (avg=10) and lead time is 5 days -> 50.0
    forecasts = [10.0, 10.0, 10.0]
    dl_extrap = calculate_lead_time_demand(forecasts, lead_time_days=5)
    assert dl_extrap == 50.0


def test_reorder_point_calculation():
    # D_L = 50.0, SS = 15.0 -> ROP = 65.0
    rop = calculate_reorder_point(lead_time_demand=50.0, safety_stock=15.0)
    assert rop == 65.0


def test_reorder_quantity_formula():
    # ROP = 100.0, S = 40.0, I = 10.0
    # Q = ceil(100 - 40 - 10) = 50
    q, expl = calculate_reorder_quantity(rop=100.0, current_stock=40.0, incoming_stock=10.0)
    assert q == 50

    # Net deficit <= 0: S = 120, ROP = 100 -> Q = 0
    q_zero, _ = calculate_reorder_quantity(rop=100.0, current_stock=120.0, incoming_stock=0.0)
    assert q_zero == 0

    # Fractional deficit: ROP = 50.5, S = 20.0, I = 0.0 -> deficit = 30.5 -> ceil is 31
    q_ceil, _ = calculate_reorder_quantity(rop=50.5, current_stock=20.0, incoming_stock=0.0)
    assert q_ceil == 31


def test_missing_stock_handling():
    # If S is None, Q must NOT be calculated authoritatively
    q, expl = calculate_reorder_quantity(rop=100.0, current_stock=None, incoming_stock=0.0)
    assert q is None
    assert "unavailable" in expl.lower()


def test_unconfirmed_incoming_stock_handling():
    # If I is None, system must NOT silently invent or subtract it; treats I = 0
    q, expl = calculate_reorder_quantity(rop=100.0, current_stock=60.0, incoming_stock=None)
    assert q == 40
    assert "unknown" in expl.lower() or "unconfirmed" in expl.lower()


def test_estimated_cost_calculation():
    # Q = 50, Unit Cost = $2.50 -> Cost = $125.00
    cost, expl = calculate_estimated_cost(quantity=50, unit_cost=2.50)
    assert cost == 125.00

    # Q = 0 -> Cost = 0.0
    cost_zero, _ = calculate_estimated_cost(quantity=0, unit_cost=2.50)
    assert cost_zero == 0.0

    # Unit cost missing (None) -> Cost is None
    cost_none, expl_none = calculate_estimated_cost(quantity=50, unit_cost=None)
    assert cost_none is None
    assert "unavailable" in expl_none.lower()


def test_risk_classification():
    # High: S (30) < D_L (50), ROP (70)
    level_h, _, _ = classify_inventory_risk(current_stock=30.0, lead_time_demand=50.0, rop=70.0)
    assert level_h == "High"

    # Medium: D_L (50) <= S (60) < ROP (70)
    level_m, _, _ = classify_inventory_risk(current_stock=60.0, lead_time_demand=50.0, rop=70.0)
    assert level_m == "Medium"

    # Low: S (80) >= ROP (70)
    level_l, _, _ = classify_inventory_risk(current_stock=80.0, lead_time_demand=50.0, rop=70.0)
    assert level_l == "Low"

    # Unknown: S is None
    level_u, _, _ = classify_inventory_risk(current_stock=None, lead_time_demand=50.0, rop=70.0)
    assert level_u == "Unknown"


def test_days_of_cover():
    # S = 100, daily demand = 20 -> DOC = 5.0
    doc = calculate_days_of_cover(current_stock=100.0, daily_demand=20.0)
    assert doc == 5.0

    # Missing stock -> None
    assert calculate_days_of_cover(current_stock=None, daily_demand=20.0) is None

    # Zero demand -> None (avoid division by zero)
    assert calculate_days_of_cover(current_stock=100.0, daily_demand=0.0) is None
