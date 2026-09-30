"""One contract for every harness: start a task, stream events, get a typed result.

Both harnesses emit the same event and result types, so the application can
pick a harness per request without rebuilding its task lifecycle around each
runtime. Failures are structured (code + message + retryable), never bare
exceptions leaking out of the harness.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

_ids = itertools.count(1)

EventKind = Literal["started", "decision", "plan", "tool_call", "tool_result", "blocked",
                    "verify", "question", "escalated", "completed", "failed"]
Status = Literal["completed", "needs_user", "escalated", "failed"]


@dataclass
class TaskRequest:
    text: str
    state: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: f"task-{next(_ids)}")


@dataclass
class TaskEvent:
    task_id: str
    harness: str
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        body = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.harness}] {self.kind:<11} {body}"


@dataclass
class Failure:
    code: Literal["budget_exhausted", "tool_error", "policy_denied", "no_progress", "invalid_output"]
    message: str
    retryable: bool = False


@dataclass
class TaskResult:
    task_id: str
    harness: str
    status: Status
    output: dict[str, Any] = field(default_factory=dict)
    events: list[TaskEvent] = field(default_factory=list)
    failure: Failure | None = None
