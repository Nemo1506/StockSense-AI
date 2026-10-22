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


# ---------------------------------------------------------------------------
# CSV PERSISTENCE FOR REPLENISHMENT ORDERS & INVENTORY UPDATES
# ---------------------------------------------------------------------------

PRODUCT_NAME_TO_SKU: Dict[str, str] = {
    "Wireless Earbuds": "P0005",
    "Running Shoes": "P0008",
    "Protein Powder": "P0001",
    "Jeans": "P0012",
    "Coffee Beans": "P0006",
    "T-shirts": "P0014",
    "Smartwatch": "P0009",
    "Office Chair": "P0007",
    "Action Figure": "P0002",
    "Yoga Mat": "P0003"
}


def _resolve_product_id(item_name: str, sku: str) -> str:
    """Resolves arbitrary product names / mock SKUs into Kaggle Product IDs (P0001 - P0020)."""
    if sku and sku.startswith("P0") and len(sku) == 5:
        return sku
    for name, mapped_sku in PRODUCT_NAME_TO_SKU.items():
        if name.lower() in str(item_name).lower() or name.lower() in str(sku).lower():
            return mapped_sku
    return sku or "P0001"


def commit_replenishment_to_csv(
    recommendations: Any,
    csv_path: str = "retail_store_inventory.csv",
    orders_csv_path: str = "data/replenishment_orders.csv",
    authorized_by: str = "Autonomous Replenishment System"
) -> Dict[str, Any]:
    """
    Persists replenishment order quantities directly into the master retail CSV dataset.
    1. Updates 'Units Ordered' on the latest date records for each Store ID and Product ID in retail_store_inventory.csv.
    2. Logs each transaction to data/replenishment_orders.csv for historical auditability.
    Returns a summary dictionary of updated records and ledger entries.
    """
    import os
    import pandas as pd
    from datetime import datetime

    items_to_commit = []

    # Parse different input structures: DataFrame, list of objects, list of dicts
    if hasattr(recommendations, "iterrows"):
        for _, row in recommendations.iterrows():
            auth = row.get("Authorize", True)
            if auth is False:
                continue
            raw_sku = str(row.get("SKU", ""))
            raw_prod = str(row.get("Product", ""))
            pid = _resolve_product_id(raw_prod, raw_sku)
            raw_store = str(row.get("Store", row.get("Store ID", "S001")))
            if "(" in raw_prod and ")" in raw_prod:
                try:
                    raw_store = raw_prod.split("(")[1].split(")")[0].strip()
                except Exception:
                    pass
            q = int(row.get("Proposed Q", row.get("Units Ordered", 0)) or 0)
            u_cost = float(row.get("Unit Cost ($)", row.get("Unit Cost", row.get("unit_cost", 3.50))) or 3.50)
            t_cost = float(row.get("Order Cost ($)", row.get("Order Cost", row.get("estimated_cost", q * u_cost))) or (q * u_cost))
            cat = str(row.get("Category", "General"))
            if q > 0:
                items_to_commit.append({
                    "product_id": pid,
                    "store_id": raw_store,
                    "product_name": raw_prod or pid,
                    "category": cat,
                    "quantity": q,
                    "unit_cost": u_cost,
                    "total_cost": round(t_cost, 2)
                })
    elif isinstance(recommendations, list):
        for r in recommendations:
            if hasattr(r, "proposed_quantity"):
                q = getattr(r, "proposed_quantity", 0)
                if q and q > 0:
                    pid = _resolve_product_id(getattr(r, "product_id", ""), getattr(r, "product_id", ""))
                    items_to_commit.append({
                        "product_id": pid,
                        "store_id": getattr(r, "store_id", "S001"),
                        "product_name": getattr(r, "product_id", ""),
                        "category": getattr(r, "category", "General"),
                        "quantity": int(q),
                        "unit_cost": float(getattr(r, "unit_cost", 3.50) or 3.50),
                        "total_cost": float(getattr(r, "estimated_cost", 0.0) or (int(q) * 3.50))
                    })
            elif isinstance(r, dict):
                q = int(r.get("quantity", r.get("proposed_quantity", r.get("Units Ordered", 0))) or 0)
                if q > 0:
                    raw_p = str(r.get("product_id", r.get("Product", r.get("SKU", ""))))
                    pid = _resolve_product_id(raw_p, raw_p)
                    items_to_commit.append({
                        "product_id": pid,
                        "store_id": str(r.get("store_id", r.get("Store", "S001"))),
                        "product_name": raw_p,
                        "category": str(r.get("category", "General")),
                        "quantity": q,
                        "unit_cost": float(r.get("unit_cost", 3.50) or 3.50),
                        "total_cost": float(r.get("total_cost", q * 3.50) or (q * 3.50))
                    })

    if not items_to_commit:
        return {
            "success": True,
            "updated_rows": 0,
            "logged_orders": 0,
            "message": "No replenishment items with Q > 0 were provided."
        }

    updated_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_today = datetime.now().strftime("%Y-%m-%d")

    # 1. Update retail_store_inventory.csv directly
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "Date" in df.columns and "Product ID" in df.columns:
                max_date = str(df["Date"].max())
                for item in items_to_commit:
                    pid = item["product_id"]
                    sid = item["store_id"]
                    q = item["quantity"]

                    # Target latest date for this store and product
                    mask = (df["Date"].astype(str) == max_date) & (df["Product ID"] == pid)
                    if "Store ID" in df.columns and sid in df["Store ID"].values:
                        store_mask = mask & (df["Store ID"] == sid)
                        if store_mask.any():
                            mask = store_mask

                    if not mask.any():
                        # Fallback to the latest available record for this product
                        prod_rows = df[df["Product ID"] == pid]
                        if not prod_rows.empty:
                            p_max = str(prod_rows["Date"].max())
                            mask = (df["Date"].astype(str) == p_max) & (df["Product ID"] == pid)
                            if "Store ID" in df.columns and sid in prod_rows["Store ID"].values:
                                store_mask = mask & (df["Store ID"] == sid)
                                if store_mask.any():
                                    mask = store_mask

                    if mask.any() and "Units Ordered" in df.columns:
                        df.loc[mask, "Units Ordered"] = q
                        updated_count += int(mask.sum())

                df.to_csv(csv_path, index=False)
        except Exception as e:
            pass

    # 2. Append to persistent replenishment orders ledger
    logged_count = 0
    try:
        orders_dir = os.path.dirname(orders_csv_path)
        if orders_dir and not os.path.exists(orders_dir):
            os.makedirs(orders_dir, exist_ok=True)

        order_rows = []
        for i, it in enumerate(items_to_commit):
            order_id = f"PO-{datetime.now().strftime('%Y%m%d')}-{i+1:04d}"
            order_rows.append({
                "Order_ID": order_id,
                "Date": date_today,
                "Store_ID": it["store_id"],
                "Product_ID": it["product_id"],
                "Product_Name": it["product_name"],
                "Category": it["category"],
                "Units_Ordered": it["quantity"],
                "Unit_Cost": it["unit_cost"],
                "Total_Cost": it["total_cost"],
                "Status": "AUTHORIZED",
                "Authorized_By": authorized_by,
                "Timestamp": now_str
            })

        df_orders = pd.DataFrame(order_rows)
        header_needed = not os.path.exists(orders_csv_path) or os.path.getsize(orders_csv_path) == 0
        df_orders.to_csv(orders_csv_path, mode="a", header=header_needed, index=False)
        logged_count = len(order_rows)
    except Exception as e:
        pass

    return {
        "success": True,
        "updated_rows": updated_count,
        "logged_orders": logged_count,
        "authorized_by": authorized_by,
        "timestamp": now_str,
        "csv_path": csv_path,
        "orders_csv_path": orders_csv_path
    }


def commit_single_item_to_csv(
    product_id: str,
    product_name: str,
    store_id: str = "S001",
    category: str = "General",
    units_ordered: int = 100,
    unit_cost: float = 3.50,
    csv_path: str = "retail_store_inventory.csv",
    orders_csv_path: str = "data/replenishment_orders.csv",
    authorized_by: str = "Manual Override"
) -> Dict[str, Any]:
    """Helper to commit a single product replenishment order directly to CSV."""
    item = {
        "product_id": product_id,
        "product_name": product_name,
        "store_id": store_id,
        "category": category,
        "quantity": units_ordered,
        "unit_cost": unit_cost,
        "total_cost": round(units_ordered * unit_cost, 2)
    }
    return commit_replenishment_to_csv(
        recommendations=[item],
        csv_path=csv_path,
        orders_csv_path=orders_csv_path,
        authorized_by=authorized_by
    )

