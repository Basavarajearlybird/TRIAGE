"""Request/response types passed between the experiment scripts and the router."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None          # small-model id; None => router auto-picks
    system: Optional[str] = None
    max_signals: bool = True             # compute the deep signals, not just uncertainty
    allow_escalation: bool = True
    escalate_to: Optional[str] = None    # big model to escalate to


class SignalSet(BaseModel):
    uncertainty: float = 0.0
    instability: float = 0.0
    contradiction: float = 0.0
    retrieval_disagreement: float = 0.0
    evidence_sufficiency: float = 1.0
    detail: dict[str, Any] = Field(default_factory=dict)


class CostMetrics(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0
    llm_calls: int = 0
    # A free tier hides compute cost, it does not remove it. compute_units is a
    # hardware-agnostic proxy (decode weighted 10x prefill); routing_overhead_passes
    # counts the extra model passes the signal machinery costs beyond one baseline pass.
    compute_units: float = 0.0
    routing_overhead_passes: int = 0


class RouteDecision(BaseModel):
    tier: int
    tier_name: str
    long_context: bool = False
    retrieved: bool = False
    verified: bool = False
    escalated: bool = False
    escalated_to: Optional[str] = None
    models_used: list[str] = Field(default_factory=list)
    prefilter_route: str = "normal"      # easy_direct | normal | hard_direct | tool
    predicted_difficulty: Optional[float] = None
    abstained: bool = False
    abstain_risk: float = 0.0
    reason: str = ""


class ChatResponse(BaseModel):
    answer: str
    status: Literal["OK", "PENDING_REVIEW"] = "OK"
    model: str
    # Which provider actually served the final answer. "mock" means the run did
    # not reach a live API, so its numbers are not a real measurement.
    provider: str
    route: RouteDecision
    signals: SignalSet
    cost: CostMetrics
    latency_ms: float = 0.0
