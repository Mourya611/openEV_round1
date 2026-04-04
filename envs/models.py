from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    ticket_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    issue_type: Literal["billing", "bug", "security", "integration", "compliance"]
    message: str
    urgency_hint: Literal["low", "medium", "high", "critical"]


class TriageAction(BaseModel):
    ticket_id: str = Field(..., description="Ticket id to handle.")
    decision: Literal["resolve", "escalate", "request_info", "defer"]
    priority: Literal["low", "medium", "high", "urgent"]
    response_template: Literal["short", "empathetic", "technical", "compliance"]
    notes: str = Field(..., min_length=4, max_length=240)


class RewardModel(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    components: Dict[str, float] = Field(default_factory=dict)


class ObservationModel(BaseModel):
    task_name: str
    objective: str
    current_ticket: Optional[Ticket] = None
    queue_remaining: int = 0
    processed_count: int = 0
    progress: float = Field(..., ge=0.0, le=1.0)
    last_feedback: str = ""
    allowed_decisions: List[str] = Field(
        default_factory=lambda: ["resolve", "escalate", "request_info", "defer"]
    )
    allowed_priorities: List[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "urgent"]
    )
    allowed_templates: List[str] = Field(
        default_factory=lambda: ["short", "empathetic", "technical", "compliance"]
    )


class StepResult(BaseModel):
    observation: ObservationModel
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


class EpisodeState(BaseModel):
    task_name: str
    objective: str
    max_steps: int
    step_count: int
    current_index: int
    done: bool
    total_reward: float
    tickets: List[Ticket]
    actions_taken: List[TriageAction]
    rewards: List[float]
