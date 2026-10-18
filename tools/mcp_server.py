"""
tools/mcp_server.py
Optional Model Context Protocol (MCP) Server for StockSense AI.
Exposes read-only deterministic tools over stdio / JSON-RPC:
1. inspect_inventory_data
2. calculate_replenishment_math
3. retrieve_inventory_policy
4. validate_replenishment_safety
"""

import sys
import json
import os
from typing import Dict, Any, List

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.inventory_tools import (
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_reorder_quantity,
    calculate_estimated_cost,
    classify_inventory_risk,
    calculate_days_of_cover
)
from tools.policy_retriever import PolicyRetriever


TOOLS_MANIFEST = [
    {
        "name": "inspect_inventory_data",
        "description": "Reads and profiles a CSV inventory file, returning column names and row count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to CSV file"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "calculate_replenishment_math",
        "description": "Calculates D_L, ROP, Q, and Estimated Cost deterministically using retail replenishment formulas.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "daily_forecasts": {"type": "array", "items": {"type": "number"}, "description": "Forecasted daily sales"},
                "lead_time_days": {"type": "integer", "description": "Supplier lead time in days"},
                "safety_stock": {"type": "number", "description": "Configured safety stock units"},
                "current_stock": {"type": "number", "description": "Current on-hand stock"},
                "incoming_stock": {"type": "number", "description": "Confirmed in-transit inventory", "default": 0},
                "unit_cost": {"type": "number", "description": "Unit procurement cost"}
            },
            "required": ["daily_forecasts", "lead_time_days", "safety_stock", "current_stock"]
        }
    },
    {
        "name": "retrieve_inventory_policy",
        "description": "Performs TF-IDF retrieval of relevant corporate replenishment policy clauses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Inventory topic or situation keyword"},
                "top_k": {"type": "integer", "description": "Number of clauses to return", "default": 2}
            },
            "required": ["query"]
        }
    },
    {
        "name": "validate_replenishment_safety",
        "description": "Independently recalculates replenishment numbers to verify safety and consistency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reported_rop": {"type": "number"},
                "reported_q": {"type": "integer"},
                "lead_time_demand": {"type": "number"},
                "safety_stock": {"type": "number"},
                "current_stock": {"type": "number"},
                "incoming_stock": {"type": "number", "default": 0}
            },
            "required": ["reported_rop", "reported_q", "lead_time_demand", "safety_stock", "current_stock"]
        }
    }
]


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "inspect_inventory_data":
        path = arguments.get("file_path", "data/sample_inventory.csv")
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        import pandas as pd
        df = pd.read_csv(path)
        return {
            "row_count": len(df),
            "columns": list(df.columns),
            "preview": df.head(3).to_dict(orient="records")
        }

    elif tool_name == "calculate_replenishment_math":
        daily = arguments.get("daily_forecasts", [])
        l = int(arguments.get("lead_time_days", 7))
        ss = float(arguments.get("safety_stock", 14))
        s = float(arguments.get("current_stock", 0))
        i = float(arguments.get("incoming_stock", 0))
        cost_unit = arguments.get("unit_cost")

        d_l = calculate_lead_time_demand(daily, l)
        rop = calculate_reorder_point(d_l, ss)
        q, q_note = calculate_reorder_quantity(rop, s, i)
        cost, cost_note = calculate_estimated_cost(q, cost_unit)
        risk, risk_reason, _ = classify_inventory_risk(s, d_l, rop)

        return {
            "lead_time_demand": d_l,
            "reorder_point": rop,
            "reorder_quantity": q,
            "quantity_explanation": q_note,
            "estimated_cost": cost,
            "cost_explanation": cost_note,
            "risk_level": risk,
            "risk_reason": risk_reason
        }

    elif tool_name == "retrieve_inventory_policy":
        retriever = PolicyRetriever()
        results = retriever.retrieve(arguments.get("query", ""), top_k=arguments.get("top_k", 2))
        return {"results": results}

    elif tool_name == "validate_replenishment_safety":
        rep_rop = arguments.get("reported_rop")
        rep_q = arguments.get("reported_q")
        dl = arguments.get("lead_time_demand")
        ss = arguments.get("safety_stock")
        s = arguments.get("current_stock")
        i = arguments.get("incoming_stock", 0)

        expected_rop = round(dl + ss, 2)
        expected_q, _ = calculate_reorder_quantity(expected_rop, s, i)

        is_rop_valid = abs(expected_rop - rep_rop) < 0.05
        is_q_valid = (expected_q == rep_q)
        verdict = "PASS" if (is_rop_valid and is_q_valid) else "BLOCKED"

        return {
            "verdict": verdict,
            "expected_rop": expected_rop,
            "reported_rop": rep_rop,
            "expected_q": expected_q,
            "reported_q": rep_q,
            "is_consistent": (is_rop_valid and is_q_valid)
        }

    return {"error": f"Unknown tool: {tool_name}"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("StockSense AI MCP Server Manifest:")
        print(json.dumps(TOOLS_MANIFEST, indent=2))
        sample_call = handle_tool_call(
            "calculate_replenishment_math",
            {
                "daily_forecasts": [10.0, 10.0, 10.0],
                "lead_time_days": 3,
                "safety_stock": 10.0,
                "current_stock": 25.0,
                "incoming_stock": 0.0,
                "unit_cost": 4.50
            }
        )
        print("\nTest Tool Execution Result:")
        print(json.dumps(sample_call, indent=2))
        sys.exit(0)

    # Standard MCP JSON-RPC Stdio loop
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")

            if method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_MANIFEST}}
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                out = handle_tool_call(tool_name, args)
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(out)}]}}
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {method} not found"}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
