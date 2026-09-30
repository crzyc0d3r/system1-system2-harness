"""System 2 harness: the model plans; application code keeps authority.

Loop:  plan -> guarded tool call -> result back into state -> verify -> repeat

Each cycle ends in exactly one of: another tool call, a final action, or a
question for the user. The planner (an LLM in production) only *proposes*
steps. The GuardedToolRouter decides whether a proposed call may run
(allowlist, per-tool quotas, approval-required tools), and the harness owns
the budgets, stop conditions and final execution.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from .protocol import Failure, TaskEvent, TaskRequest, TaskResult


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    requires_approval: bool = False  # side-effecting tools: harness asks the user first
    max_calls: int = 5


@dataclass
class Step:
    kind: Literal["tool", "final", "ask"]
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    thought: str = ""


class Planner(Protocol):
    def next_step(self, goal: str, constraints: list[str], state: dict[str, Any],
                  tools: list[Tool]) -> Step: ...


class GuardedToolRouter:
    """The only path from a planned step to a real side effect."""

    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.calls: dict[str, int] = {}

    def check(self, step: Step, approved: set[str]) -> str | None:
        """Return None if the call may run, else a denial reason."""
        tool = self.tools.get(step.tool or "")
        if tool is None:
            return f"tool {step.tool!r} is not on the allowlist"
        if self.calls.get(tool.name, 0) >= tool.max_calls:
            return f"tool {tool.name!r} exceeded its quota of {tool.max_calls} calls"
        if tool.requires_approval and tool.name not in approved:
            return f"tool {tool.name!r} requires user approval"
        return None

    def call(self, step: Step) -> Any:
        tool = self.tools[step.tool]  # type: ignore[index]
        self.calls[tool.name] = self.calls.get(tool.name, 0) + 1
        return tool.fn(**step.args)


Verifier = Callable[[dict[str, Any]], "str | None"]  # returns a progress key, or None


class AgentHarness:
    name = "system2"

    def __init__(self, planner: Planner, tools: list[Tool], constraints: list[str] | None = None,
                 max_steps: int = 8, max_stalls: int = 2, verifier: Verifier | None = None):
        self.planner = planner
        self.tools = tools
        self.constraints = constraints or []
        self.max_steps = max_steps
        self.max_stalls = max_stalls
        self.verifier = verifier or (lambda s: json.dumps(s.get("facts", {}), sort_keys=True, default=str))

    def run(self, req: TaskRequest, approved: set[str] | None = None) -> TaskResult:
        approved = approved or set()
        router = GuardedToolRouter(self.tools)
        state: dict[str, Any] = {"facts": {}, "history": [], **req.state}
        ev = [TaskEvent(req.task_id, self.name, "started", {"goal": req.text})]
        last_progress, stalls = self.verifier(state), 0

        def done(status, output, failure=None):
            kind = "failed" if status == "failed" else ("question" if status == "needs_user" else "completed")
            ev.append(TaskEvent(req.task_id, self.name, kind, output if not failure else {"error": failure.code}))
            return TaskResult(req.task_id, self.name, status, {**output, "state": state}, ev, failure)

        for i in range(self.max_steps):
            step = self.planner.next_step(req.text, self.constraints, state, self.tools)
            ev.append(TaskEvent(req.task_id, self.name, "plan",
                                {"step": i + 1, "kind": step.kind, "tool": step.tool, "thought": step.thought}))
            if step.kind == "final":
                return done("completed", {"answer": step.message})
            if step.kind == "ask":
                return done("needs_user", {"question": step.message})

            denial = router.check(step, approved)
            if denial:
                ev.append(TaskEvent(req.task_id, self.name, "blocked", {"reason": denial}))
                state["history"].append({"tool": step.tool, "denied": denial})
                tool = router.tools.get(step.tool or "")
                if tool is not None and tool.requires_approval:
                    # Code, not the model, turns a gated side effect into a user question.
                    return done("needs_user", {
                        "question": f"May I run {tool.name} with {step.args}?",
                        "pending": {"tool": tool.name, "args": step.args},
                    })
                continue

            ev.append(TaskEvent(req.task_id, self.name, "tool_call", {"tool": step.tool, "args": step.args}))
            try:
                result = router.call(step)
            except Exception as exc:  # tool failures come back as data, not crashes
                result = {"error": f"{type(exc).__name__}: {exc}"}
            ev.append(TaskEvent(req.task_id, self.name, "tool_result", {"tool": step.tool, "result": result}))
            state["history"].append({"tool": step.tool, "args": step.args, "result": result})
            state["facts"][step.tool] = result

            progress = self.verifier(state)
            stalls = stalls + 1 if progress == last_progress else 0
            last_progress = progress
            ev.append(TaskEvent(req.task_id, self.name, "verify", {"stalls": stalls}))
            if stalls >= self.max_stalls:
                return done("failed", {}, Failure("no_progress", f"{stalls} steps without new facts"))

        return done("failed", {}, Failure("budget_exhausted", f"hit max_steps={self.max_steps}", retryable=True))


class ScriptedPlanner:
    """Deterministic planner: a function of (goal, state) -> Step.

    Stands in for an LLM so the control flow can be run and tested offline.
    """

    def __init__(self, policy: Callable[[str, dict[str, Any]], Step]):
        self.policy = policy

    def next_step(self, goal, constraints, state, tools) -> Step:
        return self.policy(goal, state)


class OpenAIPlanner:
    """Optional LLM planner (any OpenAI-compatible endpoint).

    The model must answer with JSON: {"kind": "tool"|"final"|"ask", "tool": ..., "args": {...},
    "message": ..., "thought": ...}. Invalid JSON becomes an "ask" step, so bad output can
    never trigger a tool call.
    """

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # optional dependency

        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.model = model or os.getenv("PLANNER_MODEL", "gpt-4.1-mini")

    def next_step(self, goal, constraints, state, tools) -> Step:
        tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        prompt = (
            f"Goal: {goal}\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
            + f"\nTools:\n{tool_desc}\nState: {json.dumps(state, default=str)[:6000]}\n"
            "Reply with ONE JSON object: kind (tool|final|ask), tool, args, message, thought."
        )
        resp = self.client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            d = json.loads(resp.choices[0].message.content or "{}")
            return Step(kind=d["kind"], tool=d.get("tool"), args=d.get("args") or {},
                        message=d.get("message", ""), thought=d.get("thought", ""))
        except (KeyError, ValueError, TypeError):
            return Step(kind="ask", message="I could not plan the next step. Can you rephrase?")
