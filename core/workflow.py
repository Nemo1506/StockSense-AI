"""
core/workflow.py
LangGraph Orchestration for StockSense AI.
Coordinates Planner, Forecaster, Inventory Analyst, Replenishment Agent, and Critic Validator.
Implements shared state, conditional routing, bounded retry (max 1), and explicit logs.
"""

from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import math
import numpy as np
import pandas as pd

from langgraph.graph import StateGraph, END

from core.data_validation import ColumnMapping, DataProfile, profile_and_validate_dataset
from core.planner import PlannerAgent, ExecutionPlan, PlanStep
from core.forecaster import DemandForecastingAgent, ForecastResult
from core.inventory_agent import InventoryRiskAgent, RiskAssessment
from core.replenishment_agent import ReplenishmentAgent, ReplenishmentRecommendation
from core.critic_agent import CriticAgent, ValidationResult
from tools.policy_retriever import PolicyRetriever


class WorkflowState(TypedDict):
    user_request: str
    raw_df: pd.DataFrame
    column_mapping: ColumnMapping
    user_config: Dict[str, Any]
    data_profile: Optional[DataProfile]
    clean_df: Optional[pd.DataFrame]
    execution_plan: Optional[ExecutionPlan]
    forecast_results: Dict[str, ForecastResult]
    risk_results: Dict[str, RiskAssessment]
    recommendations: List[ReplenishmentRecommendation]
    validation_result: Optional[ValidationResult]
    retry_count: int
    logs: List[Dict[str, str]]
    errors: List[str]
    workflow_status: str  # STARTED, PLANNING, DATA_ERROR, FORECASTING, ASSESSING, REPLENISHING, VALIDATING, COMPLETED, BLOCKED, HUMAN_REVIEW


def log_event(state: WorkflowState, agent: str, message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    state["logs"].append({
        "timestamp": timestamp,
        "agent": agent,
        "level": level,
        "message": message
    })


# --- NODES ---

def planner_node(state: WorkflowState) -> WorkflowState:
    """Agent 1: Planner Node - validates data readiness and designs execution plan."""
    enabled = state.get("user_config", {}).get("agents_enabled", {}).get("planner", True)
    if not enabled:
        log_event(state, "Planner / Coordinator", "Planner Agent is manually deactivated by operator. Using direct execution baseline.", "WARNING")
        profile, clean_df = profile_and_validate_dataset(
            df=state["raw_df"],
            mapping=state["column_mapping"]
        )
        state["data_profile"] = profile
        state["clean_df"] = clean_df
        state["execution_plan"] = ExecutionPlan(
            plan_id="PLAN-MANUAL-OVERRIDE",
            objective="Autonomous direct execution baseline with planner bypassed.",
            steps=[
                PlanStep(step_id=1, name="Data Validation", agent_assigned="Data Validator", tools_required=[], status="COMPLETED"),
                PlanStep(step_id=2, name="Demand Forecasting", agent_assigned="Demand Forecaster", tools_required=["7-SMA", "Ridge"], status="PENDING"),
                PlanStep(step_id=3, name="Inventory Risk Analysis", agent_assigned="Inventory Risk Agent", tools_required=["ROP Formula"], status="PENDING"),
                PlanStep(step_id=4, name="Replenishment Formulation", agent_assigned="Replenishment Agent", tools_required=["Q Formula", "Policy RAG"], status="PENDING"),
                PlanStep(step_id=5, name="Critic Validation", agent_assigned="Critic Validator", tools_required=["Invariant Auditor"], status="PENDING")
            ],
            data_quality_verdict="VALID",
            missing_prerequisites=[],
            assumptions_applied={"planner_bypassed": True},
            can_proceed=True,
            llm_rationale="Manual operator override: Planner bypassed by user toggle."
        )
        state["workflow_status"] = "PLANNING"
        return state

    log_event(state, "Planner / Coordinator", "Initiating planning and dataset validation pre-flight checks.")

    # 1. Profile and validate dataset
    profile, clean_df = profile_and_validate_dataset(
        df=state["raw_df"],
        mapping=state["column_mapping"]
    )
    state["data_profile"] = profile
    state["clean_df"] = clean_df

    if profile.validation_status == "INVALID":
        state["errors"].extend(profile.errors)
        state["workflow_status"] = "DATA_ERROR"
        log_event(state, "Planner / Coordinator", f"Dataset validation failed: {'; '.join(profile.errors)}", "ERROR")
        return state

    # 2. Construct Execution Plan
    planner = PlannerAgent(
        ollama_url=state["user_config"].get("ollama_url", "http://localhost:11434"),
        ollama_model=state["user_config"].get("ollama_model", "llama3")
    )
    plan = planner.create_plan(
        user_request=state["user_request"],
        profile=profile,
        mapping=state["column_mapping"],
        user_config=state["user_config"]
    )
    state["execution_plan"] = plan

    if not plan.can_proceed:
        state["errors"].extend(plan.missing_prerequisites)
        state["workflow_status"] = "DATA_ERROR"
        log_event(state, "Planner / Coordinator", f"Execution plan blocked: missing prerequisites.", "WARNING")
        return state

    log_event(state, "Planner / Coordinator", f"Plan established: {plan.plan_id}. Proceeding to forecasting.")
    state["workflow_status"] = "PLANNING"
    return state


def forecaster_node(state: WorkflowState) -> WorkflowState:
    """Agent 2: Demand Forecaster Node - aggregates sales and predicts demand."""
    enabled = state.get("user_config", {}).get("agents_enabled", {}).get("forecaster", True)
    if not enabled:
        log_event(state, "Demand Forecaster", "Demand Forecaster is deactivated by operator. Generating autonomous mean sales baselines.", "WARNING")
        clean_df = state.get("clean_df")
        horizon_days = state.get("user_config", {}).get("horizon_days", 7)
        fallback_fc: Dict[str, ForecastResult] = {}
        if clean_df is not None and not clean_df.empty:
            for (prod, store), grp in clean_df.groupby(["_product_id", "_store_id"]):
                sorted_grp = grp.sort_values("_parsed_date")
                sales = sorted_grp["_parsed_sales"].dropna().tolist()
                dates = sorted_grp["_parsed_date"].dropna().dt.strftime("%Y-%m-%d").tolist()
                avg_daily = float(sum(sales) / len(sales)) if sales else 15.0
                last_dt = pd.to_datetime(dates[-1]) if dates else pd.to_datetime("2024-01-01")
                fc_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=horizon_days).strftime("%Y-%m-%d").tolist()
                fc_sales = [round(avg_daily, 2)] * horizon_days
                key = f"{prod}::{store}"
                fallback_fc[key] = ForecastResult(
                    product_id=str(prod),
                    store_id=str(store),
                    historical_dates=dates,
                    historical_sales=[float(s) for s in sales],
                    forecast_dates=fc_dates,
                    forecast_sales=fc_sales,
                    model_name="Autonomous Baseline (Forecaster Bypassed)",
                    metrics={"MAE": 0.0, "RMSE": 0.0, "WAPE": 0.0},
                    average_daily_demand=round(avg_daily, 2)
                )
        state["forecast_results"] = fallback_fc
        log_event(state, "Demand Forecaster", f"Generated {len(fallback_fc)} autonomous baseline series.")
        state["workflow_status"] = "FORECASTING"
        return state

    log_event(state, "Demand Forecaster", "Executing chronological baseline forecasting (7-SMA & Holdout).")
    clean_df = state["clean_df"]

    model_choice = state["user_config"].get("forecast_model", "7-SMA")
    horizon_days = state["user_config"].get("horizon_days", 7)

    forecaster = DemandForecastingAgent(
        forecast_model_choice=model_choice,
        horizon_days=horizon_days
    )

    forecast_dict = forecaster.run_all(clean_df=clean_df)
    state["forecast_results"] = forecast_dict

    log_event(state, "Demand Forecaster", f"Generated forecasts for {len(forecast_dict)} product/store series.")
    state["workflow_status"] = "FORECASTING"
    return state


def inventory_risk_node(state: WorkflowState) -> WorkflowState:
    """Agent 3: Inventory Risk Node - computes D_L, DOC, and stratifies stockout risk."""
    enabled = state.get("user_config", {}).get("agents_enabled", {}).get("inventory_risk", True)
    if not enabled:
        log_event(state, "Inventory Risk Analyst", "Inventory Risk Analyst is deactivated by operator. Computing autonomous ROP baselines.", "WARNING")
        lead_time = state.get("user_config", {}).get("lead_time_days", 7)
        safety_stock = state.get("user_config", {}).get("safety_stock", 14.0)
        clean_df = state.get("clean_df")
        default_stock = state.get("user_config", {}).get("default_stock", 40.0)
        has_inv_col = clean_df is not None and "_parsed_inventory" in clean_df.columns

        fallback_risks: Dict[str, RiskAssessment] = {}
        for key, fc in state.get("forecast_results", {}).items():
            cur_stock = default_stock
            if has_inv_col:
                sub = clean_df[(clean_df["_product_id"] == fc.product_id) & (clean_df["_store_id"] == fc.store_id)]
                if not sub.empty:
                    last_val = sub.sort_values("_parsed_date")["_parsed_inventory"].dropna()
                    if not last_val.empty:
                        cur_stock = float(last_val.iloc[-1])

            d_l = round(fc.average_daily_demand * lead_time, 2)
            rop = round(d_l + safety_stock, 2)
            doc = round(cur_stock / fc.average_daily_demand, 1) if (cur_stock is not None and fc.average_daily_demand > 0) else None
            risk_lvl = "High" if (cur_stock is not None and cur_stock < d_l) else ("Medium" if (cur_stock is not None and cur_stock < rop) else "Low")

            fallback_risks[key] = RiskAssessment(
                product_id=fc.product_id,
                store_id=fc.store_id,
                current_stock=cur_stock,
                lead_time_days=lead_time,
                safety_stock=safety_stock,
                lead_time_demand=d_l,
                reorder_point=rop,
                days_of_cover=doc,
                risk_level=risk_lvl,
                risk_reason=f"Autonomous risk baseline (Risk Agent bypassed): S={cur_stock if cur_stock is not None else 0:.0f}, ROP={rop:.1f}, D_L={d_l:.1f}.",
                evidence={"lead_time_demand": d_l, "reorder_point": rop, "current_stock": cur_stock, "bypassed": True}
            )
        state["risk_results"] = fallback_risks
        log_event(state, "Inventory Risk Analyst", f"Established {len(fallback_risks)} autonomous inventory risk positions.")
        state["workflow_status"] = "ASSESSING"
        return state

    log_event(state, "Inventory Risk Analyst", "Analyzing inventory coverage and stockout risk against lead time.")
    lead_time = state["user_config"].get("lead_time_days", 7)
    safety_stock = state["user_config"].get("safety_stock", 14.0)

    analyst = InventoryRiskAgent(lead_time_days=lead_time, safety_stock=safety_stock)

    # Build current stock lookup from clean_df or user default
    stock_lookup = {}
    clean_df = state["clean_df"]
    has_inv_col = "_parsed_inventory" in clean_df.columns
    default_stock = state["user_config"].get("default_stock")

    for key, fc in state["forecast_results"].items():
        if has_inv_col:
            # Extract latest stock recorded for this product & store
            sub = clean_df[
                (clean_df["_product_id"] == fc.product_id) &
                (clean_df["_store_id"] == fc.store_id)
            ]
            if not sub.empty:
                last_stock = sub.sort_values("_parsed_date")["_parsed_inventory"].dropna().iloc[-1]
                stock_lookup[key] = float(last_stock)
            else:
                stock_lookup[key] = default_stock
        else:
            stock_lookup[key] = default_stock

    risk_dict = analyst.run_all(
        forecast_results=state["forecast_results"],
        stock_lookup=stock_lookup
    )
    state["risk_results"] = risk_dict

    high_risks = sum(1 for r in risk_dict.values() if r.risk_level == "High")
    med_risks = sum(1 for r in risk_dict.values() if r.risk_level == "Medium")
    log_event(state, "Inventory Risk Analyst", f"Risk stratification: {high_risks} High, {med_risks} Medium.")
    state["workflow_status"] = "ASSESSING"
    return state


def replenishment_node(state: WorkflowState) -> WorkflowState:
    """Agent 4: Replenishment Node - calculates reorder quantity Q and queries policy RAG."""
    enabled = state.get("user_config", {}).get("agents_enabled", {}).get("replenishment", True)
    if not enabled:
        log_event(state, "Replenishment Agent", "Replenishment Agent is deactivated by operator. Formulating mathematical baseline reorders.", "WARNING")
        clean_df = state.get("clean_df")
        default_cost = state.get("user_config", {}).get("default_unit_cost", 3.50)
        default_incoming = state.get("user_config", {}).get("default_incoming_stock", 0.0)

        fallback_recs: List[ReplenishmentRecommendation] = []
        for rank_idx, (key, risk) in enumerate(state.get("risk_results", {}).items()):
            s = risk.current_stock if risk.current_stock is not None else 40.0
            rop = risk.reorder_point
            net_def = rop - s - default_incoming
            q = max(0, int(math.ceil(net_def))) if net_def > 0 else 0
            cost = round(q * default_cost, 2)
            cat = "General"
            if clean_df is not None and "_parsed_category" in clean_df.columns:
                sub = clean_df[clean_df["_product_id"] == risk.product_id]
                if not sub.empty:
                    cat = str(sub["_parsed_category"].iloc[-1])

            rec = ReplenishmentRecommendation(
                item_key=key,
                product_id=risk.product_id,
                store_id=risk.store_id,
                risk_level=risk.risk_level,
                current_stock=s,
                incoming_stock=default_incoming,
                lead_time_demand=risk.lead_time_demand,
                safety_stock=risk.safety_stock,
                reorder_point=rop,
                proposed_quantity=q,
                quantity_explanation=f"Autonomous mathematical fallback (Replenishment Agent bypassed): Q=max(0, ceil({rop:.1f} - {s:.0f})) = {q}.",
                unit_cost=default_cost,
                estimated_cost=cost,
                cost_explanation=f"Cost: {q} units * ${default_cost:.2f} = ${cost:.2f}.",
                urgency_rank=rank_idx + 1,
                policy_evidence=[{"citation": "Section 3.1 - Baseline Safety Stock Protocol", "content": "Autonomous fallback reorder formula Q=max(0, ROP-S-I)."}],
                advisory_action=f"Order {q} units (Replenishment Agent Bypassed)" if q > 0 else "Maintain current stock position",
                category=cat
            )
            fallback_recs.append(rec)

        priority_map = {"High": 1, "Medium": 2, "Low": 3}
        fallback_recs.sort(key=lambda r: (priority_map.get(r.risk_level, 4), -(r.proposed_quantity or 0)))
        for idx, r in enumerate(fallback_recs):
            r.urgency_rank = idx + 1

        state["recommendations"] = fallback_recs
        log_event(state, "Replenishment Agent", f"Formulated {len(fallback_recs)} baseline recommendations.")
        state["workflow_status"] = "REPLENISHING"
        return state

    log_event(state, "Replenishment Agent", "Computing proposed replenishment quantities and retrieving policy rules.")
    policy_retriever = PolicyRetriever()
    agent = ReplenishmentAgent(policy_retriever=policy_retriever)

    # Build incoming stock, unit cost, and category lookups
    inc_lookup = {}
    cost_lookup = {}
    cat_lookup = {}
    clean_df = state["clean_df"]
    default_cost = state["user_config"].get("default_unit_cost")
    default_incoming = state["user_config"].get("default_incoming_stock", 0.0)

    for key, risk in state["risk_results"].items():
        sub = clean_df[
            (clean_df["_product_id"] == risk.product_id) &
            (clean_df["_store_id"] == risk.store_id)
        ]
        if sub.empty:
            sub = clean_df[clean_df["_product_id"] == risk.product_id]

        # 1. Incoming Stock
        if not sub.empty and "_parsed_incoming" in sub.columns:
            last_inc = sub.sort_values("_parsed_date")["_parsed_incoming"].dropna()
            inc_lookup[key] = float(last_inc.iloc[-1]) if not last_inc.empty else default_incoming
        else:
            inc_lookup[key] = default_incoming

        # 2. Unit Cost / Price
        if not sub.empty and "_parsed_price" in sub.columns:
            last_p = sub.sort_values("_parsed_date")["_parsed_price"].dropna()
            cost_lookup[key] = float(last_p.iloc[-1]) if not last_p.empty else default_cost
        elif not sub.empty and state["column_mapping"].price_col in sub.columns:
            p = sub[state["column_mapping"].price_col].dropna()
            cost_lookup[key] = float(p.iloc[-1]) if not p.empty else default_cost
        else:
            cost_lookup[key] = default_cost

        # 3. Category
        if not sub.empty and "_parsed_category" in sub.columns:
            cat_lookup[key] = str(sub["_parsed_category"].iloc[-1])
        elif not sub.empty and state["column_mapping"].category_col in sub.columns:
            cat_lookup[key] = str(sub[state["column_mapping"].category_col].iloc[-1])
        else:
            cat_lookup[key] = "General"

    recs = agent.run_all(
        risk_assessments=state["risk_results"],
        incoming_stock_lookup=inc_lookup,
        unit_cost_lookup=cost_lookup,
        category_lookup=cat_lookup
    )
    state["recommendations"] = recs

    log_event(state, "Replenishment Agent", f"Formulated {len(recs)} ranked advisory replenishment proposals.")
    state["workflow_status"] = "REPLENISHING"
    return state


def critic_node(state: WorkflowState) -> WorkflowState:
    """Agent 5: Critic Node - independently recalculates math, checks policies, and issues verdict."""
    enabled = state.get("user_config", {}).get("agents_enabled", {}).get("critic", True)
    if not enabled:
        log_event(state, "Critic / Safety Validator", "Critic Guard is manually deactivated by operator. Validation audit bypassed (auto-authorized).", "WARNING")
        item_verdicts = {}
        for rec in state.get("recommendations", []):
            item_verdicts[rec.item_key] = {
                "status": "PASS",
                "reasons": ["Critic agent bypassed by operator: auto-authorized."],
                "recomputed_q": rec.proposed_quantity,
                "recomputed_cost": rec.estimated_cost
            }
        state["validation_result"] = ValidationResult(
            overall_status="PASS",
            item_verdicts=item_verdicts,
            correction_needed=False,
            correction_target=None,
            audit_notes=["Critic Guard deactivated: Zero-trust audit bypassed by operator. All recommendations auto-approved."]
        )
        state["workflow_status"] = "COMPLETED"
        return state

    log_event(state, "Critic / Safety Validator", "Running independent recalculation and policy compliance validation.")
    critic = CriticAgent(
        high_cost_review_threshold=state["user_config"].get("high_cost_threshold", 10000.0)
    )

    verdict = critic.validate_all(
        recommendations=state["recommendations"],
        correction_attempt=state["retry_count"]
    )
    state["validation_result"] = verdict

    for note in verdict.audit_notes:
        log_event(state, "Critic / Safety Validator", note)

    if verdict.overall_status == "PASS":
        state["workflow_status"] = "COMPLETED"
    elif verdict.overall_status == "REVIEW":
        state["workflow_status"] = "HUMAN_REVIEW"
    elif verdict.overall_status == "BLOCKED":
        if verdict.correction_needed and state["retry_count"] < 1:
            state["workflow_status"] = "CORRECTION"
        else:
            state["workflow_status"] = "BLOCKED"

    return state


def correction_node(state: WorkflowState) -> WorkflowState:
    """Bounded correction node: adjusts calculation discrepancies flagged by Critic (max 1 retry)."""
    state["retry_count"] += 1
    log_event(
        state,
        "Replenishment Agent",
        f"Automated correction cycle #{state['retry_count']}: recalculating flagged items.",
        "WARNING"
    )

    # Re-run replenishment with updated recalculation
    state = replenishment_node(state)
    return state


# --- CONDITIONAL ROUTING ---

def route_after_planner(state: WorkflowState) -> str:
    if state["workflow_status"] == "DATA_ERROR":
        return "END"
    return "forecaster"


def route_after_critic(state: WorkflowState) -> str:
    if state["workflow_status"] == "CORRECTION":
        return "correction"
    return "END"


# --- BUILD STATE GRAPH ---

def create_stocksense_graph() -> StateGraph:
    builder = StateGraph(WorkflowState)

    builder.add_node("planner", planner_node)
    builder.add_node("forecaster", forecaster_node)
    builder.add_node("inventory_risk", inventory_risk_node)
    builder.add_node("replenishment", replenishment_node)
    builder.add_node("critic", critic_node)
    builder.add_node("correction", correction_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "forecaster": "forecaster",
            "END": END
        }
    )

    builder.add_edge("forecaster", "inventory_risk")
    builder.add_edge("inventory_risk", "replenishment")
    builder.add_edge("replenishment", "critic")

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "correction": "correction",
            "END": END
        }
    )

    # Correction node loops back to critic
    builder.add_edge("correction", "critic")

    return builder.compile()


def stream_stocksense_workflow(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    user_config: Dict[str, Any],
    user_request: str = "Analyze inventory stockout risks and recommend replenishment quantities"
):
    """
    Generator yielding (node_name, current_state) after each agent node execution
    in the LangGraph StateGraph pipeline.
    """
    initial_state: WorkflowState = {
        "user_request": user_request,
        "raw_df": df,
        "column_mapping": mapping,
        "user_config": user_config,
        "data_profile": None,
        "clean_df": None,
        "execution_plan": None,
        "forecast_results": {},
        "risk_results": {},
        "recommendations": [],
        "validation_result": None,
        "retry_count": 0,
        "logs": [],
        "errors": [],
        "workflow_status": "STARTED"
    }

    graph = create_stocksense_graph()
    for step in graph.stream(initial_state):
        for node_name, state in step.items():
            yield node_name, state


def run_stocksense_workflow(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    user_config: Dict[str, Any],
    user_request: str = "Analyze inventory stockout risks and recommend replenishment quantities"
) -> WorkflowState:
    """
    Main entry point for running the end-to-end multi-agent LangGraph pipeline synchronously.
    """
    initial_state: WorkflowState = {
        "user_request": user_request,
        "raw_df": df,
        "column_mapping": mapping,
        "user_config": user_config,
        "data_profile": None,
        "clean_df": None,
        "execution_plan": None,
        "forecast_results": {},
        "risk_results": {},
        "recommendations": [],
        "validation_result": None,
        "retry_count": 0,
        "logs": [],
        "errors": [],
        "workflow_status": "STARTED"
    }

    graph = create_stocksense_graph()
    final_state = graph.invoke(initial_state)
    return final_state

