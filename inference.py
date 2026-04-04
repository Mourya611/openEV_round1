from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "support-ticket-triage-openenv"
SUCCESS_SCORE_THRESHOLD = 0.70
MAX_STEPS = 12


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    action_str = json.dumps(action, ensure_ascii=True, separators=(",", ":"))
    err = "null" if error is None else json.dumps(error, ensure_ascii=True)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.4f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = "[" + ",".join(f"{r:.4f}" for r in rewards) + "]"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _fallback_action(ticket: Dict[str, Any]) -> Dict[str, Any]:
    issue = ticket.get("issue_type", "")
    urgency = ticket.get("urgency_hint", "medium")

    if issue in {"security", "compliance"}:
        decision = "escalate"
        template = "compliance"
    elif issue == "bug":
        decision = "escalate"
        template = "technical"
    elif issue == "integration":
        decision = "request_info"
        template = "technical"
    else:
        decision = "resolve"
        template = "empathetic"

    priority_map = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "critical": "urgent",
    }
    priority = priority_map.get(urgency, "medium")

    return {
        "ticket_id": ticket.get("ticket_id"),
        "decision": decision,
        "priority": priority,
        "response_template": template,
        "notes": "Initial triage based on urgency, issue type, and policy.",
    }


def get_model_action(
    client: OpenAI, task_name: str, observation: Dict[str, Any], history: List[str]
) -> Dict[str, Any]:
    ticket = observation.get("current_ticket")
    if ticket is None:
        return {
            "ticket_id": "NONE",
            "decision": "defer",
            "priority": "low",
            "response_template": "short",
            "notes": "No active ticket available.",
        }

    prompt = {
        "task_name": task_name,
        "objective": observation.get("objective"),
        "current_ticket": ticket,
        "progress": observation.get("progress"),
        "history": history[-4:],
        "required_action_schema": {
            "ticket_id": "string",
            "decision": "resolve|escalate|request_info|defer",
            "priority": "low|medium|high|urgent",
            "response_template": "short|empathetic|technical|compliance",
            "notes": "string <= 240 chars",
        },
    }

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a deterministic support triage policy. Return only JSON.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    )
    raw = completion.choices[0].message.content or ""
    action = _extract_json(raw)

    required = {"ticket_id", "decision", "priority", "response_template", "notes"}
    if not required.issubset(set(action.keys())):
        raise ValueError("Model output missing required action fields.")
    return action


async def run_task(client: OpenAI, task_name: str) -> Tuple[float, bool]:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient(timeout=60.0) as http:
        await http.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name})
        result = (await http.get(f"{ENV_BASE_URL}/state")).json()

        for step in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            current_ticket = None
            idx = result.get("current_index", 0)
            tickets = result.get("tickets", [])
            if idx < len(tickets):
                current_ticket = tickets[idx]

            observation = {
                "objective": result.get("objective"),
                "current_ticket": current_ticket,
                "progress": (idx / len(tickets)) if tickets else 1.0,
            }

            error = None
            try:
                action = get_model_action(client, task_name, observation, history)
            except Exception as e:
                error = f"llm_error: {e}"
                action = _fallback_action(current_ticket or {})

            step_response = await http.post(f"{ENV_BASE_URL}/step", json=action)
            body = step_response.json()

            reward = float(body.get("reward", 0.0) or 0.0)
            done = bool(body.get("done", False))
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

            history.append(
                f"Step {step} -> ticket={action.get('ticket_id')} decision={action.get('decision')} reward={reward:.2f}"
            )

            if done:
                break

            result = (await http.get(f"{ENV_BASE_URL}/state")).json()

    if rewards:
        score = sum(rewards) / len(rewards)
    score = max(0.0, min(1.0, score))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score, success


async def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    async with httpx.AsyncClient(timeout=30.0) as http:
        tasks_resp = await http.get(f"{ENV_BASE_URL}/tasks")
        tasks_data = tasks_resp.json().get("tasks", {})
        task_names = list(tasks_data.keys())
        if len(task_names) < 3:
            raise RuntimeError("Environment must expose at least 3 tasks.")

    aggregate: List[float] = []
    for task_name in task_names:
        score, _ = await run_task(client, task_name)
        aggregate.append(score)

    mean_score = sum(aggregate) / len(aggregate) if aggregate else 0.0
    print(
        f"[END] success={str(mean_score >= SUCCESS_SCORE_THRESHOLD).lower()} steps={len(aggregate)} score={mean_score:.4f} rewards={[round(s, 4) for s in aggregate]}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
