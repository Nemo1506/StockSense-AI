"""
core/planner.py
Agent 1: Planner / Coordinator
Decomposes user inventory request, assesses data readiness, verifies prerequisites,
maintains shared execution state, and generates a structured, schema-validated execution plan.
Includes optional Ollama plan enrichment with deterministic fallback.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import requests

from core.data_validation import DataProfile, ColumnMapping


class PlanStep(BaseModel):
    step_id: int
    name: str
    agent_assigned: str
    tools_required: List[str]
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, SKIPPED, BLOCKED
    prerequisites_met: bool = True
    notes: str = ""


class ExecutionPlan(BaseModel):
    plan_id: str
    objective: str
    steps: List[PlanStep]
    data_quality_verdict: str
    missing_prerequisites: List[str] = Field(default_factory=list)
    assumptions_applied: Dict[str, Any] = Field(default_factory=dict)
    can_proceed: bool = True
    llm_rationale: Optional[str] = None


class PlannerAgent:
    """
    Agent 1: Coordinates the multi-agent replenishment pipeline.
    Decomposes the task into structured, dependency-checked stages.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "llama3"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Queries local Ollama if running. Never crashes if unavailable."""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=2.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
        except Exception:
            return None
        return None

    def create_plan(
        self,
        user_request: str,
        profile: DataProfile,
        mapping: ColumnMapping,
        user_config: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Decomposes inventory request and constructs schema-validated execution plan.
        """
        plan_id = "PLAN-SENSE-001"
        missing_prereqs = []
        assumptions = {}

        # Check required fields
        if profile.validation_status == "INVALID":
            return ExecutionPlan(
                plan_id=plan_id,
                objective=user_request,
                steps=[],
                data_quality_verdict="INVALID",
                missing_prerequisites=profile.errors,
                can_proceed=False,
                llm_rationale="Planning terminated: Dataset failed essential structural validation checks."
            )

        # Inventory stock check
        has_stock_col = mapping.inventory_col is not None
        stock_assumption = user_config.get("default_stock")
        if not has_stock_col and stock_assumption is None:
            missing_prereqs.append("Current available stock (S) missing from dataset and no default assumed.")
        elif not has_stock_col:
            assumptions["stock_level_source"] = f"User configured default assumption ({stock_assumption} units)"
        else:
            assumptions["stock_level_source"] = f"CSV column '{mapping.inventory_col}'"

        # Lead time check
        lead_time = user_config.get("lead_time_days", 7)
        assumptions["lead_time_days"] = lead_time

        # Safety stock check
        safety_stock = user_config.get("safety_stock", 14)
        assumptions["safety_stock"] = safety_stock

        # Unit cost check
        has_price_col = mapping.price_col is not None
        cost_assumption = user_config.get("default_unit_cost")
        if not has_price_col and cost_assumption is None:
            assumptions["unit_cost_status"] = "Missing from CSV and UI: Order cost will remain unspecified."
        elif cost_assumption is not None:
            assumptions["unit_cost_status"] = f"User configured assumption: ${cost_assumption:.2f}/unit."
        else:
            assumptions["unit_cost_status"] = f"CSV column '{mapping.price_col}'."

        steps = [
            PlanStep(
                step_id=1,
                name="Data Validation & Aggregation",
                agent_assigned="Planner / Data Validator",
                tools_required=["profile_and_validate_dataset", "ColumnMapping"],
                status="COMPLETED",
                prerequisites_met=True,
                notes=f"Processed {profile.total_rows} rows across {profile.unique_products_count} products."
            ),
            PlanStep(
                step_id=2,
                name="Chronological Demand Forecasting",
                agent_assigned="Demand Forecasting Agent",
                tools_required=["split_chronological", "forecast_7day_moving_average", "calculate_wape"],
                status="PENDING",
                prerequisites_met=True,
                notes="Chronological train/holdout split; 7-SMA baseline + secondary regression."
            ),
            PlanStep(
                step_id=3,
                name="Inventory Risk Stratification",
                agent_assigned="Inventory Risk Analysis Agent",
                tools_required=["calculate_lead_time_demand", "classify_inventory_risk", "calculate_days_of_cover"],
                status="PENDING",
                prerequisites_met=len(missing_prereqs) == 0,
                notes="Compute D_L over lead time and classify High/Med/Low/Unknown stockout risk."
            ),
            PlanStep(
                step_id=4,
                name="Replenishment Recommendation & Policy Grounding",
                agent_assigned="Replenishment & Business Action Agent",
                tools_required=["calculate_reorder_quantity", "calculate_estimated_cost", "PolicyRetriever"],
                status="PENDING",
                prerequisites_met=len(missing_prereqs) == 0,
                notes="Calculate Q = max(0, ceil(ROP - S - I)), query corporate SOP rules for evidence."
            ),
            PlanStep(
                step_id=5,
                name="Critic Verification & Safety Guardrails",
                agent_assigned="Critic / Safety / Validation Agent",
                tools_required=["recalculate_deterministic_math", "validate_evidence_relevance", "check_injection_defense"],
                status="PENDING",
                prerequisites_met=True,
                notes="Independent recalculation, bound retry loop (max 1 retry), PASS/REVIEW/BLOCKED decision."
            )
        ]

        # Optional LLM rationale synthesis
        llm_prompt = (
            f"You are the Planner Agent for StockSense AI. Briefly explain the execution plan for this dataset: "
            f"Products={profile.unique_products_count}, Stores={profile.unique_stores_count}, "
            f"Date Coverage={profile.date_coverage_days} days. Lead time={lead_time} days. "
            f"Keep explanation under 3 sentences."
        )
        llm_rationale = self._query_ollama(llm_prompt)
        if not llm_rationale:
            llm_rationale = (
                f"Deterministic Plan Established: Executing 5-stage sequential workflow for "
                f"{profile.unique_products_count} product(s) across {profile.unique_stores_count} store(s). "
                f"Evaluation window leverages chronological holdout with {lead_time}-day lead time demand horizon."
            )

        return ExecutionPlan(
            plan_id=plan_id,
            objective=user_request,
            steps=steps,
            data_quality_verdict=profile.validation_status,
            missing_prerequisites=missing_prereqs,
            assumptions_applied=assumptions,
            can_proceed=len(missing_prereqs) == 0,
            llm_rationale=llm_rationale
        )
