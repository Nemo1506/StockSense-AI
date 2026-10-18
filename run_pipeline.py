"""
run_pipeline.py
Entrypoint to execute the autonomous multi-agent StockSense AI workflow
via compiled LangGraph StateGraph directly from the terminal.
"""

import sys
import os
import pandas as pd

# Ensure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.data_validation import auto_detect_columns
from core.workflow import run_stocksense_workflow

def main():
    print("=" * 70)
    print("🚀 StockSense AI — Autonomous Multi-Agent Replenishment Pipeline")
    print("=" * 70)

    dataset_path = "retail_store_inventory.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/sample_inventory.csv"

    print(f"\n📂 Loading dataset from: {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded {len(df):,} records successfully.")

    # Auto-detect column mapping
    mapping = auto_detect_columns(df)
    print(f"\n🔍 Schema Auto-Detection:")
    print(f"   • Date Column:      {mapping.date_col}")
    print(f"   • Product Column:   {mapping.product_id_col}")
    print(f"   • Store Column:     {mapping.store_id_col}")
    print(f"   • Sales Column:     {mapping.sales_col}")
    print(f"   • Inventory Column: {mapping.inventory_col}")
    print(f"   • Price Column:     {mapping.price_col}")

    user_config = {
        "lead_time_days": 7,
        "safety_stock": 14,
        "forecast_horizon": 14,
        "cost_threshold": 10000.0,
        "policy_queries": [
            "reorder point minimum requirements",
            "safety stock guidelines",
            "bulk order discount"
        ]
    }

    print("\n⚙️ Active Supply Chain Parameters:")
    print(f"   • Supplier Lead Time (L): {user_config['lead_time_days']} days")
    print(f"   • Safety Stock (SS):      {user_config['safety_stock']} units")
    print(f"   • Forecast Horizon (H):   {user_config['forecast_horizon']} days")
    print(f"   • Human Review Threshold: ${user_config['cost_threshold']:,.2f}")

    print("\n🧠 Compiling & Invoking LangGraph StateGraph...")
    # Run over top 1,500 records for fast CLI benchmarking
    sample_size = min(len(df), 1500)
    final_state = run_stocksense_workflow(
        df=df.head(sample_size),
        mapping=mapping,
        user_config=user_config,
        user_request="Perform full inventory risk audit and generate replenishment orders."
    )

    print("\n" + "=" * 70)
    print("📊 LANGGRAPH EXECUTION TRACE & MULTI-AGENT LOGS")
    print("=" * 70)
    for log in final_state.get("logs", []):
        agent = log.get("agent", "SYSTEM")
        lvl = log.get("level", "INFO")
        msg = log.get("message", "")
        print(f"  [{agent:<15}] [{lvl}] {msg}")

    print("\n" + "=" * 70)
    print("📋 AUTONOMOUS REPLENISHMENT PURCHASE ORDERS")
    print("=" * 70)
    recs = final_state.get("recommendations", [])
    if not recs:
        print("  No purchase orders needed; all active stock levels are currently healthy.")
    else:
        print(f"  Total Recommended POs: {len(recs)}")
        print(f"  {'SKU':<12} | {'Q (Units)':<10} | {'Priority':<9} | {'Est. Cost ($)':<14} | {'Policy Grounding'}")
        print("  " + "-" * 80)
        for r in recs[:8]:
            sku = getattr(r, "product_id", getattr(r, "item_key", "N/A"))
            store = getattr(r, "store_id", "S001")
            q = getattr(r, "proposed_quantity", 0)
            prio = getattr(r, "risk_level", "Low")
            cost = getattr(r, "estimated_cost", 0.0)
            cit = "SOP-INV-2026-V3"
            cost_str = f"${cost:,.2f}" if cost is not None else "N/A"
            item_display = f"{sku} ({store})"
            print(f"  {item_display:<12} | {str(q):<10} | {prio:<9} | {cost_str:<14} | {cit}")

    print("\n" + "=" * 70)
    print("🛡️ ZERO-TRUST CRITIC AUDIT & GOVERNANCE VERDICT")
    print("=" * 70)
    val = final_state.get("validation_result")
    if val:
        status = getattr(val, "overall_status", "UNKNOWN")
        arith_issues = getattr(val, "arithmetic_discrepancies", [])
        review_issues = getattr(val, "review_flags", getattr(val, "issues", []))
        print(f"  Final Audit Verdict:      {status}")
        print(f"  Arithmetic Discrepancies: {len(arith_issues)}")
        print(f"  Governance Review Flags:  {len(review_issues)}")
        print(f"  Cyclical Retry Loops:     {final_state.get('retry_count', 0)}")
        if arith_issues:
            print(f"  Arithmetic Mismatches:")
            for iss in arith_issues[:3]:
                print(f"   • {iss}")
        if review_issues:
            print(f"  Governance Escalation Details:")
            for iss in review_issues[:3]:
                print(f"   • {iss}")
    else:
        print("  Validation result pending.")

    print("\n" + "=" * 70)
    print("✅ Programme Execution Complete!")
    print("👉 View Interactive UI at: http://localhost:8501")
    print("=" * 70)

if __name__ == "__main__":
    main()
