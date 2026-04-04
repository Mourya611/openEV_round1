from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import Ticket


@dataclass(frozen=True)
class TicketRubric:
    decision: str
    priority: str
    response_template: str
    must_include_keywords: List[str]


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    difficulty: str
    objective: str
    max_steps: int
    tickets: List[Ticket]
    rubric_by_ticket: Dict[str, TicketRubric]


def get_tasks() -> Dict[str, TaskDefinition]:
    easy_tickets = [
        Ticket(
            ticket_id="E-1001",
            customer_tier="pro",
            issue_type="billing",
            message="I was charged twice for the same monthly invoice.",
            urgency_hint="medium",
        ),
        Ticket(
            ticket_id="E-1002",
            customer_tier="enterprise",
            issue_type="security",
            message="Our SOC flagged suspicious login attempts from unknown countries.",
            urgency_hint="critical",
        ),
    ]
    medium_tickets = [
        Ticket(
            ticket_id="M-2001",
            customer_tier="free",
            issue_type="bug",
            message="The export button fails silently in Chrome 125.",
            urgency_hint="high",
        ),
        Ticket(
            ticket_id="M-2002",
            customer_tier="pro",
            issue_type="integration",
            message="Slack integration keeps disconnecting every two hours.",
            urgency_hint="medium",
        ),
        Ticket(
            ticket_id="M-2003",
            customer_tier="enterprise",
            issue_type="compliance",
            message="Need DPA wording clarification for EU onboarding this week.",
            urgency_hint="high",
        ),
    ]
    hard_tickets = [
        Ticket(
            ticket_id="H-3001",
            customer_tier="enterprise",
            issue_type="security",
            message="Possible data exfiltration, logs indicate unusual API key usage.",
            urgency_hint="critical",
        ),
        Ticket(
            ticket_id="H-3002",
            customer_tier="pro",
            issue_type="billing",
            message="Contracted annual discount was not applied in renewal invoice.",
            urgency_hint="high",
        ),
        Ticket(
            ticket_id="H-3003",
            customer_tier="free",
            issue_type="integration",
            message="Webhook retries are delayed and causing duplicate events.",
            urgency_hint="medium",
        ),
        Ticket(
            ticket_id="H-3004",
            customer_tier="enterprise",
            issue_type="compliance",
            message="Customer requests SOC2 evidence and subprocessor list in 24h.",
            urgency_hint="critical",
        ),
    ]

    return {
        "ticket-triage-easy": TaskDefinition(
            name="ticket-triage-easy",
            difficulty="easy",
            objective="Prioritize and route common billing and security issues correctly.",
            max_steps=4,
            tickets=easy_tickets,
            rubric_by_ticket={
                "E-1001": TicketRubric(
                    decision="resolve",
                    priority="medium",
                    response_template="empathetic",
                    must_include_keywords=["refund", "invoice"],
                ),
                "E-1002": TicketRubric(
                    decision="escalate",
                    priority="urgent",
                    response_template="compliance",
                    must_include_keywords=["security", "incident"],
                ),
            },
        ),
        "ticket-triage-medium": TaskDefinition(
            name="ticket-triage-medium",
            difficulty="medium",
            objective="Handle mixed bug/integration/compliance tickets with accurate routing.",
            max_steps=6,
            tickets=medium_tickets,
            rubric_by_ticket={
                "M-2001": TicketRubric(
                    decision="escalate",
                    priority="high",
                    response_template="technical",
                    must_include_keywords=["reproduce", "browser"],
                ),
                "M-2002": TicketRubric(
                    decision="request_info",
                    priority="medium",
                    response_template="technical",
                    must_include_keywords=["logs", "timestamps"],
                ),
                "M-2003": TicketRubric(
                    decision="escalate",
                    priority="high",
                    response_template="compliance",
                    must_include_keywords=["legal", "dpa"],
                ),
            },
        ),
        "ticket-triage-hard": TaskDefinition(
            name="ticket-triage-hard",
            difficulty="hard",
            objective="Perform policy-safe triage under high urgency and conflicting constraints.",
            max_steps=8,
            tickets=hard_tickets,
            rubric_by_ticket={
                "H-3001": TicketRubric(
                    decision="escalate",
                    priority="urgent",
                    response_template="compliance",
                    must_include_keywords=["containment", "security"],
                ),
                "H-3002": TicketRubric(
                    decision="request_info",
                    priority="high",
                    response_template="empathetic",
                    must_include_keywords=["contract", "discount"],
                ),
                "H-3003": TicketRubric(
                    decision="escalate",
                    priority="medium",
                    response_template="technical",
                    must_include_keywords=["idempotency", "webhook"],
                ),
                "H-3004": TicketRubric(
                    decision="escalate",
                    priority="urgent",
                    response_template="compliance",
                    must_include_keywords=["soc2", "subprocessor"],
                ),
            },
        ),
    }
