"""Application-level router: System 1 first, System 2 when the path is open-ended.

Most production apps need both. Cheap, bounded judgments go through System 1;
anything it escalates (low confidence, policy block, or an intent that needs
several dependent actions) is handed to System 2 with the same TaskRequest.
"""
from __future__ import annotations

from .protocol import TaskEvent, TaskRequest, TaskResult
from .system1 import DecisionHarness
from .system2 import AgentHarness


class AppRouter:
    def __init__(self, fast: DecisionHarness, slow: AgentHarness):
        self.fast, self.slow = fast, slow

    def handle(self, req: TaskRequest, approved: set[str] | None = None) -> TaskResult:
        first = self.fast.run(req)
        if first.status != "escalated":
            return first
        req.state = {**req.state, "escalation": first.output}
        second = self.slow.run(req, approved=approved)
        second.events = first.events + [
            TaskEvent(req.task_id, "router", "escalated", {"to": self.slow.name})
        ] + second.events
        return second
