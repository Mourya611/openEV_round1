from __future__ import annotations

from typing import Dict, List

from .models import TriageAction
from .tasks import TaskDefinition

OPEN_INTERVAL_EPSILON = 0.01


def _clamp_open_unit_interval(value: float) -> float:
    return max(OPEN_INTERVAL_EPSILON, min(1.0 - OPEN_INTERVAL_EPSILON, value))


def score_action(task: TaskDefinition, action: TriageAction) -> Dict[str, float]:
    rubric = task.rubric_by_ticket.get(action.ticket_id)
    if rubric is None:
        return {
            "decision_match": 0.0,
            "priority_match": 0.0,
            "template_match": 0.0,
            "keyword_coverage": 0.0,
            "total": OPEN_INTERVAL_EPSILON,
        }

    decision_match = 1.0 if action.decision == rubric.decision else 0.0
    priority_match = 1.0 if action.priority == rubric.priority else 0.0
    template_match = 1.0 if action.response_template == rubric.response_template else 0.0

    notes = action.notes.lower()
    required = rubric.must_include_keywords
    keyword_hits = sum(1 for keyword in required if keyword in notes)
    keyword_coverage = keyword_hits / len(required) if required else 1.0

    total = (
        0.40 * decision_match
        + 0.30 * priority_match
        + 0.20 * template_match
        + 0.10 * keyword_coverage
    )
    total = _clamp_open_unit_interval(total)
    return {
        "decision_match": decision_match,
        "priority_match": priority_match,
        "template_match": template_match,
        "keyword_coverage": keyword_coverage,
        "total": total,
    }


def grade_episode(task: TaskDefinition, actions: List[TriageAction]) -> float:
    if not actions:
        return OPEN_INTERVAL_EPSILON

    ticket_scores = []
    seen = set()
    for action in actions:
        if action.ticket_id in seen:
            ticket_scores.append(0.0)
            continue
        seen.add(action.ticket_id)
        ticket_scores.append(score_action(task, action)["total"])

    normalized = sum(ticket_scores) / len(task.tickets)
    return _clamp_open_unit_interval(normalized)
