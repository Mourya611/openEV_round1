from __future__ import annotations

from fastapi import FastAPI, HTTPException

from envs import SupportTicketTriageEnv
from envs.models import ResetRequest, TriageAction

app = FastAPI(title="OpenEnv Support Ticket Triage", version="0.1.0")
env = SupportTicketTriageEnv()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "support-ticket-triage"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": env.list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    task_name = None if req is None else req.task_name
    if task_name is not None and task_name not in env.tasks:
        raise HTTPException(status_code=400, detail=f"Unknown task_name: {task_name}")
    result = env.reset(task_name=task_name)
    return result.model_dump()


@app.post("/step")
def step(action: TriageAction) -> dict:
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return env.state()
