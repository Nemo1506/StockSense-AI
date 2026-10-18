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
