from __future__ import annotations

from typing import Dict, Optional

from .graders import OPEN_INTERVAL_EPSILON, grade_episode, score_action
from .models import EpisodeState, ObservationModel, StepResult, TriageAction
from .tasks import TaskDefinition, get_tasks


class SupportTicketTriageEnv:
    """
    OpenEnv-style environment for real-world support ticket triage.
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, TaskDefinition] = get_tasks()
        self.current_task: Optional[TaskDefinition] = None
        self.state_data: Optional[EpisodeState] = None
        self.reset()

    def _make_observation(self, feedback: str = "") -> ObservationModel:
        assert self.current_task is not None
        assert self.state_data is not None

        idx = self.state_data.current_index
        current_ticket = None
        if idx < len(self.state_data.tickets):
            current_ticket = self.state_data.tickets[idx]

        progress = (
            self.state_data.current_index / len(self.state_data.tickets)
            if self.state_data.tickets
            else 1.0
        )

        return ObservationModel(
            task_name=self.current_task.name,
            objective=self.current_task.objective,
            current_ticket=current_ticket,
            queue_remaining=max(0, len(self.state_data.tickets) - idx),
            processed_count=self.state_data.current_index,
            progress=max(0.0, min(1.0, progress)),
            last_feedback=feedback,
        )

    def reset(self, task_name: Optional[str] = None) -> StepResult:
        if task_name and task_name in self.tasks:
            self.current_task = self.tasks[task_name]
        elif self.current_task is None:
            # Default task for first reset call.
            self.current_task = self.tasks["ticket-triage-easy"]

        assert self.current_task is not None
        self.state_data = EpisodeState(
            task_name=self.current_task.name,
            objective=self.current_task.objective,
            max_steps=self.current_task.max_steps,
            step_count=0,
            current_index=0,
            done=False,
            total_reward=0.0,
            tickets=self.current_task.tickets,
            actions_taken=[],
            rewards=[],
        )
        observation = self._make_observation(feedback="Episode reset.")
        return StepResult(observation=observation, reward=0.0, done=False, info={"task": self.current_task.name})

    def state(self) -> Dict[str, object]:
        assert self.state_data is not None
        return self.state_data.model_dump()

    def step(self, action: TriageAction) -> StepResult:
        assert self.current_task is not None
        assert self.state_data is not None

        if self.state_data.done:
            observation = self._make_observation(feedback="Episode already complete.")
            return StepResult(
                observation=observation,
                reward=OPEN_INTERVAL_EPSILON,
                done=True,
                info={"reason": "episode_done", "final_score": grade_episode(self.current_task, self.state_data.actions_taken)},
            )

        self.state_data.step_count += 1
        idx = self.state_data.current_index
        expected_ticket = (
            self.state_data.tickets[idx] if idx < len(self.state_data.tickets) else None
        )

        feedback = ""
        reward_val = OPEN_INTERVAL_EPSILON

        if expected_ticket is None:
            self.state_data.done = True
            feedback = "No more tickets in queue."
        elif action.ticket_id != expected_ticket.ticket_id:
            # Penalize selecting wrong ticket to discourage random jumps.
            feedback = (
                f"Wrong ticket. Expected {expected_ticket.ticket_id}, got {action.ticket_id}."
            )
            reward_val = OPEN_INTERVAL_EPSILON
        else:
            components = score_action(self.current_task, action)
            action_quality = components["total"]
            projected_progress = (self.state_data.current_index + 1) / len(
                self.state_data.tickets
            )
            # Dense reward combines local action quality with global progress.
            reward_val = 0.75 * action_quality + 0.25 * projected_progress
            reward_val = max(OPEN_INTERVAL_EPSILON, min(1.0 - OPEN_INTERVAL_EPSILON, reward_val))

            self.state_data.actions_taken.append(action)
            self.state_data.rewards.append(reward_val)
            self.state_data.total_reward += reward_val
            self.state_data.current_index += 1
            feedback = (
                f"Processed {action.ticket_id} with action quality {action_quality:.2f}."
            )

        queue_done = self.state_data.current_index >= len(self.state_data.tickets)
        steps_done = self.state_data.step_count >= self.state_data.max_steps
        self.state_data.done = queue_done or steps_done

        final_score = grade_episode(self.current_task, self.state_data.actions_taken)
        info = {
            "task": self.current_task.name,
            "step_count": self.state_data.step_count,
            "max_steps": self.state_data.max_steps,
            "final_score": final_score if self.state_data.done else None,
        }
        observation = self._make_observation(feedback=feedback)

        return StepResult(
            observation=observation,
            reward=reward_val,
            done=self.state_data.done,
            info=info,
        )

    def list_tasks(self) -> Dict[str, Dict[str, object]]:
        return {
            name: {
                "difficulty": task.difficulty,
                "objective": task.objective,
                "max_steps": task.max_steps,
                "num_tickets": len(task.tickets),
            }
            for name, task in self.tasks.items()
        }
