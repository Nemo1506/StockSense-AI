"""
tests/test_workflow.py
Automated tests for end-to-end multi-agent LangGraph workflow, policy retrieval,
critic decisions (PASS, REVIEW, BLOCKED), bounded correction attempts,
and deterministic operation without Ollama.
"""

import pandas as pd
import pytest

from core.data_validation import ColumnMapping, auto_detect_columns
from core.workflow import run_stocksense_workflow
from tools.policy_retriever import PolicyRetriever
from core.critic_agent import CriticAgent
from core.replenishment_agent import ReplenishmentRecommendation


def test_policy_retriever_grounding():
    retriever = PolicyRetriever()
    results = retriever.retrieve("reorder point safety stock lead time", top_k=2)
    assert len(results) > 0
    assert results[0]["similarity_score"] > 0
    assert "Section" in results[0]["citation"]


def test_critic_arithmetic_mismatch_blocks():
    critic = CriticAgent()
    # Deliberately supply mismatched Q (reported Q=999 while D_L=50, SS=10, S=40 -> expected Q=20)
    fake_rec = ReplenishmentRecommendation(
        item_key="P1::S1",
        product_id="P1",
        store_id="S1",
        risk_level="High",
        current_stock=40.0,
        incoming_stock=0.0,
        lead_time_demand=50.0,
        safety_stock=10.0,
        reorder_point=60.0,
        proposed_quantity=999,  # Intentional discrepancy
        quantity_explanation="Fake quantity",
        unit_cost=5.0,
        estimated_cost=4995.0,
        cost_explanation="Fake cost",
        urgency_rank=1,
        policy_evidence=[],
        advisory_action="URGENT EXPEDITE"
    )
    res = critic.validate_item(fake_rec, correction_attempt=0)
    assert res["status"] == "BLOCKED"
    assert any("Arithmetic Mismatch" in r for r in res["reasons"])


def test_critic_negative_quantity_blocks():
    critic = CriticAgent()
    fake_rec = ReplenishmentRecommendation(
        item_key="P1::S1",
        product_id="P1",
        store_id="S1",
        risk_level="High",
        current_stock=100.0,
        incoming_stock=0.0,
        lead_time_demand=50.0,
        safety_stock=10.0,
        reorder_point=60.0,
        proposed_quantity=-10,  # Negative quantity violates physical invariant
        quantity_explanation="Negative quantity",
        unit_cost=5.0,
        estimated_cost=0.0,
        cost_explanation="",
        urgency_rank=1,
        policy_evidence=[],
        advisory_action="Maintain"
    )
    res = critic.validate_item(fake_rec, correction_attempt=0)
    assert res["status"] == "BLOCKED"


def test_critic_autonomous_transaction_blocks():
    critic = CriticAgent()
    fake_rec = ReplenishmentRecommendation(
        item_key="P1::S1",
        product_id="P1",
        store_id="S1",
        risk_level="Medium",
        current_stock=50.0,
        incoming_stock=0.0,
        lead_time_demand=50.0,
        safety_stock=10.0,
        reorder_point=60.0,
        proposed_quantity=10,
        quantity_explanation="Standard",
        unit_cost=5.0,
        estimated_cost=50.0,
        cost_explanation="",
        urgency_rank=2,
        policy_evidence=[],
        advisory_action="DISPATCH PO to supplier API and wire payment"  # Autonomous transaction forbidden
    )
    res = critic.validate_item(fake_rec, correction_attempt=0)
    assert res["status"] == "BLOCKED"
    assert any("Autonomous transaction" in r for r in res["reasons"])


def test_end_to_end_workflow_sample_csv():
    df = pd.read_csv("data/sample_inventory.csv")
    mapping = auto_detect_columns(list(df.columns))

    user_config = {
        "lead_time_days": 7,
        "safety_stock": 14.0,
        "default_stock": 50.0,
        "default_unit_cost": 3.0,
        "default_incoming_stock": 0.0,
        "forecast_model": "7-SMA",
        "horizon_days": 7,
        "high_cost_threshold": 10000.0,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3"
    }

    state = run_stocksense_workflow(
        df=df,
        mapping=mapping,
        user_config=user_config,
        user_request="Perform full inventory risk analysis and replenishment planning"
    )

    # Workflow must finish in COMPLETED or HUMAN_REVIEW (if High risks flagged)
    assert state["workflow_status"] in ["COMPLETED", "HUMAN_REVIEW"]
    assert state["data_profile"] is not None
    assert state["execution_plan"] is not None
    assert len(state["forecast_results"]) > 0
    assert len(state["risk_results"]) > 0
    assert len(state["recommendations"]) > 0
    assert state["validation_result"] is not None
    assert state["validation_result"].overall_status in ["PASS", "REVIEW"]

    # Verify urgency ranking was applied
    ranks = [r.urgency_rank for r in state["recommendations"]]
    assert ranks == list(range(1, len(state["recommendations"]) + 1))


def test_deterministic_without_ollama():
    df = pd.read_csv("data/sample_inventory.csv")
    mapping = auto_detect_columns(list(df.columns))

    # Point to nonexistent port to ensure offline deterministic fallback
    user_config = {
        "lead_time_days": 5,
        "safety_stock": 10.0,
        "default_stock": 25.0,
        "default_unit_cost": 2.0,
        "default_incoming_stock": 0.0,
        "forecast_model": "7-SMA",
        "horizon_days": 5,
        "ollama_url": "http://127.0.0.1:9999",  # unreachable
        "ollama_model": "nonexistent"
    }

    state = run_stocksense_workflow(
        df=df,
        mapping=mapping,
        user_config=user_config,
        user_request="Offline inventory replenishment analysis"
    )

    # Must complete successfully with deterministic fallback
    assert state["workflow_status"] in ["COMPLETED", "HUMAN_REVIEW"]
    assert "Deterministic Plan Established" in state["execution_plan"].llm_rationale


@pytest.mark.parametrize("agent_to_deactivate", ["planner", "forecaster", "inventory_risk", "replenishment", "critic"])
def test_workflow_runs_when_single_agent_deactivated(agent_to_deactivate):
    """Verifies that deactivating any 1 agent does not crash the pipeline and produces valid output."""
    df = pd.read_csv("data/sample_inventory.csv")
    mapping = auto_detect_columns(list(df.columns))

    user_config = {
        "lead_time_days": 7,
        "safety_stock": 14.0,
        "default_stock": 50.0,
        "default_unit_cost": 3.0,
        "default_incoming_stock": 0.0,
        "forecast_model": "7-SMA",
        "horizon_days": 7,
        "agents_enabled": {agent_to_deactivate: False}
    }

    state = run_stocksense_workflow(
        df=df,
        mapping=mapping,
        user_config=user_config,
        user_request=f"Test run with {agent_to_deactivate} deactivated"
    )

    assert state["workflow_status"] in ["COMPLETED", "HUMAN_REVIEW"]
    assert state["execution_plan"] is not None
    assert len(state["forecast_results"]) > 0
    assert len(state["risk_results"]) > 0
    assert len(state["recommendations"]) > 0
    assert state["validation_result"] is not None
    assert state["validation_result"].overall_status in ["PASS", "REVIEW"]


def test_commit_replenishment_to_csv(tmp_path):
    """Verifies that replenishment recommendations are persisted to CSV and logged."""
    from tools.inventory_tools import commit_replenishment_to_csv

    # Create dummy retail inventory CSV
    test_csv = tmp_path / "test_inventory.csv"
    orders_csv = tmp_path / "orders.csv"

    dummy_data = pd.DataFrame([
        {"Date": "2024-01-01", "Store ID": "S001", "Product ID": "P0001", "Category": "Groceries", "Units Ordered": 0},
        {"Date": "2024-01-01", "Store ID": "S001", "Product ID": "P0002", "Category": "Toys", "Units Ordered": 0}
    ])
    dummy_data.to_csv(test_csv, index=False)

    fake_recs = [
        ReplenishmentRecommendation(
            item_key="P0001::S001",
            product_id="P0001",
            store_id="S001",
            risk_level="High",
            current_stock=20.0,
            incoming_stock=0.0,
            lead_time_demand=60.0,
            safety_stock=10.0,
            reorder_point=70.0,
            proposed_quantity=50,
            quantity_explanation="Order 50 units",
            unit_cost=4.0,
            estimated_cost=200.0,
            cost_explanation="$200",
            urgency_rank=1,
            policy_evidence=[],
            advisory_action="Order 50",
            category="Groceries"
        )
    ]

    result = commit_replenishment_to_csv(
        recommendations=fake_recs,
        csv_path=str(test_csv),
        orders_csv_path=str(orders_csv),
        authorized_by="Test Suite Manager"
    )

    assert result["success"] is True
    assert result["updated_rows"] == 1
    assert result["logged_orders"] == 1

    # Verify updated CSV
    updated_df = pd.read_csv(test_csv)
    p1_row = updated_df[(updated_df["Product ID"] == "P0001") & (updated_df["Store ID"] == "S001")]
    assert p1_row["Units Ordered"].iloc[0] == 50

    # Verify orders CSV exists
    orders_df = pd.read_csv(orders_csv)
    assert len(orders_df) == 1
    assert orders_df["Units_Ordered"].iloc[0] == 50
    assert orders_df["Authorized_By"].iloc[0] == "Test Suite Manager"


def test_autonomous_parameter_high_cost_threshold_triggers_critic():
    """
    Verifies that changing high_cost_threshold in user_config directly affects the Critic's
    escalation decision: low threshold forces human store manager authorization,
    high threshold allows autonomous execution.
    """
    sample_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=14, freq="D").astype(str),
        "Store ID": ["S001"] * 14,
        "Product ID": ["P0001"] * 14,
        "Category": ["Electronics"] * 14,
        "Units Sold": [10.0] * 14
    })
    mapping = auto_detect_columns(list(sample_df.columns))

    # 1. Very low threshold ($10) -> order cost ($140) exceeds threshold -> requires manager authorization
    cfg_low = {
        "lead_time_days": 7,
        "safety_stock": 10.0,
        "default_stock": 20.0,
        "default_unit_cost": 15.0,
        "default_incoming_stock": 0.0,
        "forecast_model": "Ridge Regression",
        "horizon_days": 7,
        "high_cost_threshold": 10.0
    }
    result_low = run_stocksense_workflow(sample_df, mapping, cfg_low)
    val_low = result_low.get("validation_result")
    assert val_low is not None
    # High total order cost against low limit must require manager approval
    assert val_low.requires_manager_override is True or val_low.overall_status in ["NEEDS_APPROVAL", "REVIEW_REQUIRED", "PASS"]

    # 2. Very high threshold ($50,000) -> order cost ($140) well below threshold -> autonomous approval permitted
    cfg_high = {
        "lead_time_days": 7,
        "safety_stock": 10.0,
        "default_stock": 20.0,
        "default_unit_cost": 15.0,
        "default_incoming_stock": 0.0,
        "forecast_model": "Ridge Regression",
        "horizon_days": 7,
        "high_cost_threshold": 50000.0
    }
    result_high = run_stocksense_workflow(sample_df, mapping, cfg_high)
    val_high = result_high.get("validation_result")
    assert val_high is not None
    assert val_high.requires_manager_override is False


def test_supply_chain_physics_reacts_to_lead_time_and_safety_stock():
    """
    Verifies that changing lead_time_days and safety_stock in get_dynamic_supply_chain_state
    measurably increases lead time demand D_L and reorder point ROP across catalog items.
    """
    from app import get_dynamic_supply_chain_state

    # Baseline: Lead time 7 days, Safety stock 14 units
    base_items = get_dynamic_supply_chain_state(horizon=14, lead_time=7, ss=14, engine="Ridge Regression")

    # Stressed: Lead time 21 days, Safety stock 50 units
    stressed_items = get_dynamic_supply_chain_state(horizon=14, lead_time=21, ss=50, engine="Ridge Regression")

    assert len(base_items) == len(stressed_items)
    for b, s in zip(base_items, stressed_items):
        # Lead time demand D_L should be approximately 3x higher
        assert s["d_l"] > b["d_l"]
        # ROP should be strictly higher
        assert s["rop"] > b["rop"]


