---
title: Openenv Ticket Triage
emoji: "📊"
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

# OpenEnv: Support Ticket Triage Environment

## Submission links

- GitHub repository: https://github.com/Mourya611/openEV_round1
- Hugging Face Space: https://hf.co/spaces/Mourya234/openenv-ticket-triage
- Hugging Face app URL: https://mourya234-openenv-ticket-triage.hf.space

This project implements a complete real-world OpenEnv-style environment where an agent learns to triage customer support tickets.

## Why this environment

Support triage is a real operational workflow in SaaS teams. Agents must balance urgency, policy, customer context, and technical uncertainty. This environment captures that with deterministic scoring and progressive task difficulty.

## Problem modeled

- Domain: customer support operations
- Goal: choose the right triage action for each ticket
- Realism:
  - Different issue types (`billing`, `security`, `integration`, `compliance`, `bug`)
  - Tier-aware urgency context
  - Policy-sensitive escalations for security/compliance scenarios

## OpenEnv interface

The environment exposes:

- `POST /reset` -> starts a fresh episode (optionally by task name)
- `POST /step` -> applies one agent action
- `GET /state` -> returns full internal state
- `GET /tasks` -> lists task metadata

The environment core class is `SupportTicketTriageEnv` in `envs/environment.py`, with:

- `reset(task_name: Optional[str])`
- `step(action: TriageAction)`
- `state()`

Typed models are implemented via Pydantic in `envs/models.py`.

## Action space

`TriageAction` fields:

- `ticket_id: str`
- `decision: resolve | escalate | request_info | defer`
- `priority: low | medium | high | urgent`
- `response_template: short | empathetic | technical | compliance`
- `notes: str` (required rationale text)

## Observation space

`ObservationModel` includes:

- `task_name`
- `objective`
- `current_ticket` (or `null`)
- `queue_remaining`
- `processed_count`
- `progress` (reported inside `(0, 1)` to avoid boundary-value validator issues)
- `last_feedback`
- allowed decisions/priorities/templates

## Reward design

Per-step reward is reported strictly inside `(0, 1)` and combines:

- Local action quality from deterministic rubric matching:
  - decision correctness
  - priority correctness
  - response template correctness
  - keyword coverage in notes
- Global trajectory progress bonus

Formula:

`reward = 0.75 * action_quality + 0.25 * projected_progress`, clamped into `(0,1)`.

This provides dense, partial-progress signals and discourages random behavior.

## Tasks and graders

Three deterministic tasks are included:

1. `ticket-triage-easy` (2 tickets)
2. `ticket-triage-medium` (3 tickets)
3. `ticket-triage-hard` (4 tickets)

Task definitions: `envs/tasks.py`  
Graders: `envs/graders.py`  
Episode final score: normalized deterministic grade reported strictly inside `(0, 1)`.

## Inference baseline (required)

The required root script `inference.py`:

- Uses OpenAI client for all LLM calls
- Falls back to a deterministic built-in policy if the OpenAI client or token is unavailable
- Reads the validator-required env vars `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- Emits structured stdout logs:
  - `[START]`
  - `[STEP]`
  - `[END]`
- Formats rewards to 2 decimal places and emits only validator-compatible line types

## Environment variables

Create `.env` from `.env.example`:

```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-5-mini
HF_TOKEN=your_hf_token
ENV_BASE_URL=http://localhost:7860
```

`API_BASE_URL` and `MODEL_NAME` include defaults in `inference.py`. If `HF_TOKEN` is not set, the script falls back to the deterministic local policy so the validator can still execute the run.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
python inference.py
```

## Docker

```bash
docker build -t openenv-ticket-triage .
docker run --rm -p 7860:7860 openenv-ticket-triage
```

## Quick precheck

After starting the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\precheck.ps1
```

## Hugging Face Spaces deployment

1. Push this repository to GitHub.
2. Create a Docker Space on Hugging Face.
3. Connect the repo.
4. Add secrets in Space settings:
   - `API_BASE_URL` (optional if you want to override the default)
   - `MODEL_NAME`
   - `HF_TOKEN`
   - `ENV_BASE_URL` (if your Space serves the env on a non-default URL)
5. Ensure Space responds on `/health`, `/reset`, `/step`, `/state`.

## Project structure

```text
.
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── .env.example
├── envs
│   ├── __init__.py
│   ├── models.py
│   ├── tasks.py
│   ├── graders.py
│   └── environment.py
└── scripts
    └── precheck.ps1
```
