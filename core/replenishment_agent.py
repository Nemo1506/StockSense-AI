"""
core/replenishment_agent.py
Agent 4: Replenishment & Business Action Agent
Computes deterministic reorder quantities: Q = max(0, ceil(ROP - S - I))
Calculates estimated order cost where unit cost exists.
Retrieves grounded corporate policy clauses (RAG) to provide regulatory context.
Ranks proposed replenishment actions by operational urgency.
Never executes financial transactions (Advisory Decision Support only).
"""

from typing import Dict, List, Any, Optional
import pandas as pd

from core.inventory_agent import RiskAssessment
from tools.inventory_tools import calculate_reorder_quantity, calculate_estimated_cost
from tools.policy_retriever import PolicyRetriever


class ReplenishmentRecommendation:
    def __init__(
        self,
        item_key: str,
        product_id: str,
        store_id: str,
        risk_level: str,
        current_stock: Optional[float],
        incoming_stock: Optional[float],
        lead_time_demand: float,
        safety_stock: float,
        reorder_point: float,
        proposed_quantity: Optional[int],
        quantity_explanation: str,
        unit_cost: Optional[float],
        estimated_cost: Optional[float],
        cost_explanation: str,
        urgency_rank: int,
        policy_evidence: List[Dict[str, Any]],
        advisory_action: str,
        category: Optional[str] = "General"
    ):
        self.item_key = item_key
        self.product_id = product_id
        self.store_id = store_id
        self.risk_level = risk_level
        self.category = category or "General"
        self.current_stock = current_stock
        self.incoming_stock = incoming_stock
        self.lead_time_demand = lead_time_demand
        self.safety_stock = safety_stock
        self.reorder_point = reorder_point
        self.proposed_quantity = proposed_quantity
        self.quantity_explanation = quantity_explanation
        self.unit_cost = unit_cost
        self.estimated_cost = estimated_cost
        self.cost_explanation = cost_explanation
        self.urgency_rank = urgency_rank
        self.policy_evidence = policy_evidence
        self.advisory_action = advisory_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_key": self.item_key,
            "product_id": self.product_id,
            "store_id": self.store_id,
            "category": self.category,
            "risk_level": self.risk_level,
            "current_stock": self.current_stock,
            "incoming_stock": self.incoming_stock,
            "lead_time_demand": self.lead_time_demand,
            "safety_stock": self.safety_stock,
            "reorder_point": self.reorder_point,
            "proposed_quantity": self.proposed_quantity,
            "quantity_explanation": self.quantity_explanation,
            "unit_cost": self.unit_cost,
            "estimated_cost": self.estimated_cost,
            "cost_explanation": self.cost_explanation,
            "urgency_rank": self.urgency_rank,
            "policy_citations": [p["citation"] for p in self.policy_evidence],
            "advisory_action": self.advisory_action
        }


class ReplenishmentAgent:
    """
    Agent 4: Replenishment & Business Action Agent.
    """

    def __init__(self, policy_retriever: Optional[PolicyRetriever] = None):
        self.policy_retriever = policy_retriever or PolicyRetriever()

    def generate_recommendation(
        self,
        risk: RiskAssessment,
        incoming_stock: Optional[float],
        unit_cost: Optional[float],
        category: Optional[str] = "General"
    ) -> ReplenishmentRecommendation:
        """
        Calculates replenishment order and pairs with policy citations.
        """
        # Deterministic Q
        q, q_expl = calculate_reorder_quantity(
            rop=risk.reorder_point,
            current_stock=risk.current_stock,
            incoming_stock=incoming_stock
        )

        # Deterministic Cost
        cost, cost_expl = calculate_estimated_cost(quantity=q, unit_cost=unit_cost)

        # Policy Retrieval (RAG)
        is_missing_stock = risk.current_stock is None
        is_missing_cost = unit_cost is None
        is_high_val = cost is not None and cost > 10000.0

        retrieved_policies = self.policy_retriever.retrieve_for_recommendation(
            risk_level=risk.risk_level,
            is_missing_stock=is_missing_stock,
            is_missing_cost=is_missing_cost,
            is_high_value=is_high_val
        )

        # Action text
        if risk.risk_level == "High" and q and q > 0:
            action = f"URGENT EXPEDITE: Initiate supplier PO for {q} units immediately to prevent imminent stockout."
        elif risk.risk_level == "Medium" and q and q > 0:
            action = f"STANDARD CYCLE REORDER: Issue replenishment PO for {q} units in normal replenishment cycle."
        elif risk.risk_level == "Low" or q == 0:
            action = "MAINTAIN MONITORING: Current inventory is within safe bounds; zero replenishment units required."
        else:
            action = "DATA CONFIGURATION REQUIRED: Authoritative reorder quantity cannot be recommended until current stock is provided."

        item_key = f"{risk.product_id}::{risk.store_id}"

        return ReplenishmentRecommendation(
            item_key=item_key,
            product_id=risk.product_id,
            store_id=risk.store_id,
            risk_level=risk.risk_level,
            current_stock=risk.current_stock,
            incoming_stock=incoming_stock,
            lead_time_demand=risk.lead_time_demand,
            safety_stock=risk.safety_stock,
            reorder_point=risk.reorder_point,
            proposed_quantity=q,
            quantity_explanation=q_expl,
            unit_cost=unit_cost,
            estimated_cost=cost,
            cost_explanation=cost_expl,
            urgency_rank=0,  # Will be assigned during ranking
            policy_evidence=retrieved_policies,
            advisory_action=action,
            category=category
        )

    def run_all(
        self,
        risk_assessments: Dict[str, RiskAssessment],
        incoming_stock_lookup: Dict[str, Optional[float]],
        unit_cost_lookup: Dict[str, Optional[float]],
        category_lookup: Optional[Dict[str, str]] = None
    ) -> List[ReplenishmentRecommendation]:
        """
        Executes recommendations for all items and ranks them by operational urgency.
        """
        recs: List[ReplenishmentRecommendation] = []
        cat_map = category_lookup or {}

        for key, risk in risk_assessments.items():
            inc = incoming_stock_lookup.get(key)
            if inc is None:
                inc = incoming_stock_lookup.get(risk.product_id)

            cost = unit_cost_lookup.get(key)
            if cost is None:
                cost = unit_cost_lookup.get(risk.product_id)

            cat = cat_map.get(key, cat_map.get(risk.product_id, "General"))

            rec = self.generate_recommendation(
                risk=risk,
                incoming_stock=inc,
                unit_cost=cost,
                category=cat
            )
            recs.append(rec)

        # Urgency sorting:
        # High risk (1), Medium (2), Low (3), Unknown (4)
        # Then by Days of Cover ascending (None last)
        # Then by proposed quantity descending
        risk_priority = {"High": 1, "Medium": 2, "Low": 3, "Unknown": 4}

        def sort_key(item: ReplenishmentRecommendation):
            p = risk_priority.get(item.risk_level, 5)
            doc = 999999.0 if item.current_stock is None else (item.current_stock / max(1.0, item.lead_time_demand))
            qty = -(item.proposed_quantity or 0)
            return (p, doc, qty)

        recs.sort(key=sort_key)
        for rank, item in enumerate(recs, start=1):
            item.urgency_rank = rank

        return recs
