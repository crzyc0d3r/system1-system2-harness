"""System 1 harness: the model makes one bounded judgment; code owns the workflow.

The application supplies the request and the finite set of allowed outcomes.
The decider returns a typed Decision (label + confidence). Code then applies a
confidence gate and per-label policy checks before executing the mapped
action - or escalating. The model never chooses anything outside `labels`.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .protocol import Failure, TaskEvent, TaskRequest, TaskResult

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Decision:
    label: str
    confidence: float
    scores: dict[str, float]


class Decider(Protocol):
    labels: tuple[str, ...]

    def decide(self, text: str, state: dict[str, Any]) -> Decision: ...


class NaiveBayesDecider:
    """A tiny, dependency-free decision model (multinomial naive Bayes).

    It stands in for any finite-action decision model - a fine-tuned
    classifier, a small encoder, or an LLM constrained to an enum. What
    matters for the harness is the output type, not the model.
    """

    def __init__(self, examples: dict[str, list[str]], alpha: float = 1.0):
        self.labels = tuple(examples)
        self.alpha = alpha
        self.word_counts: dict[str, Counter] = defaultdict(Counter)
        self.prior: dict[str, float] = {}
        total = sum(len(v) for v in examples.values())
        vocab: set[str] = set()
        for label, texts in examples.items():
            self.prior[label] = math.log(len(texts) / total)
            for t in texts:
                toks = tokenize(t)
                self.word_counts[label].update(toks)
                vocab.update(toks)
        self.vocab_size = len(vocab)
        self.totals = {l: sum(c.values()) for l, c in self.word_counts.items()}

    def decide(self, text: str, state: dict[str, Any] | None = None) -> Decision:
        toks = tokenize(text)
        logp = {}
        for label in self.labels:
            denom = self.totals[label] + self.alpha * self.vocab_size
            logp[label] = self.prior[label] + sum(
                math.log((self.word_counts[label][t] + self.alpha) / denom) for t in toks
            )
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        probs = {l: math.exp(v - m) / z for l, v in logp.items()}
        best = max(probs, key=probs.get)
        return Decision(best, probs[best], probs)


# A policy returns None if allowed, or a reason string if the action must not run.
Policy = Callable[[TaskRequest, Decision], "str | None"]
Action = Callable[[TaskRequest, Decision], dict[str, Any]]


class DecisionHarness:
    name = "system1"

    def __init__(self, decider: Decider, actions: dict[str, Action],
                 threshold: float = 0.7, policies: dict[str, Policy] | None = None,
                 escalate_labels: tuple[str, ...] = ()):
        unknown = set(actions) - set(decider.labels)
        if unknown:
            raise ValueError(f"actions for labels the decider cannot emit: {unknown}")
        self.decider = decider
        self.actions = actions
        self.threshold = threshold
        self.policies = policies or {}
        self.escalate_labels = escalate_labels

    def run(self, req: TaskRequest) -> TaskResult:
        ev = [TaskEvent(req.task_id, self.name, "started", {"text": req.text})]
        d = self.decider.decide(req.text, req.state)
        ev.append(TaskEvent(req.task_id, self.name, "decision",
                            {"label": d.label, "confidence": round(d.confidence, 3)}))

        def escalate(reason: str) -> TaskResult:
            ev.append(TaskEvent(req.task_id, self.name, "escalated", {"reason": reason}))
            return TaskResult(req.task_id, self.name, "escalated",
                              {"decision": d.label, "reason": reason}, ev)

        if d.label not in self.decider.labels:  # defensive: typed output check
            return TaskResult(req.task_id, self.name, "failed", {}, ev,
                              Failure("invalid_output", f"unknown label {d.label!r}"))
        if d.confidence < self.threshold:
            return escalate(f"confidence {d.confidence:.2f} < {self.threshold}")
        if d.label in self.escalate_labels or d.label not in self.actions:
            return escalate(f"label {d.label!r} needs open-ended handling")
        policy = self.policies.get(d.label)
        reason = policy(req, d) if policy else None
        if reason:
            ev.append(TaskEvent(req.task_id, self.name, "blocked", {"policy": reason}))
            return escalate(f"policy: {reason}")
        out = self.actions[d.label](req, d)
        ev.append(TaskEvent(req.task_id, self.name, "completed", out))
        return TaskResult(req.task_id, self.name, "completed", {"decision": d.label, **out}, ev)
