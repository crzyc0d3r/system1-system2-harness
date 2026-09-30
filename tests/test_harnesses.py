import pytest

from harness import (AgentHarness, DecisionHarness, NaiveBayesDecider, ScriptedPlanner, Step,
                     TaskRequest, Tool)
from harness.system1 import Decision
from examples import support_desk
from examples.support_desk import EXAMPLES, build


class FixedDecider:
    labels = ("a", "b")

    def __init__(self, label, conf):
        self.d = Decision(label, conf, {})

    def decide(self, text, state):
        return self.d


def test_naive_bayes_picks_obvious_labels():
    nb = NaiveBayesDecider(EXAMPLES)
    assert nb.decide("I forgot my password").label == "password_reset"
    assert nb.decide("refund order 99").label == "refund"
    d = nb.decide("why is my invoice higher")
    assert d.label == "investigate" and abs(sum(d.scores.values()) - 1) < 1e-9


def test_system1_confidence_gate_escalates():
    h = DecisionHarness(FixedDecider("a", 0.5), {"a": lambda r, d: {"ok": 1}}, threshold=0.7)
    assert h.run(TaskRequest("x")).status == "escalated"
    h2 = DecisionHarness(FixedDecider("a", 0.9), {"a": lambda r, d: {"ok": 1}}, threshold=0.7)
    assert h2.run(TaskRequest("x")).output["ok"] == 1


def test_system1_rejects_actions_for_unknown_labels():
    with pytest.raises(ValueError):
        DecisionHarness(FixedDecider("a", 1), {"zzz": lambda r, d: {}})


def test_system1_invalid_model_output_fails_typed():
    h = DecisionHarness(FixedDecider("not-a-label", 0.99), {"a": lambda r, d: {}})
    res = h.run(TaskRequest("x"))
    assert res.status == "failed" and res.failure.code == "invalid_output"


def test_refund_policy_gate_hands_off_to_system2():
    r = build()
    small = r.handle(TaskRequest("please refund my last order", {"amount": 40}))
    assert small.harness == "system1" and small.output["action"] == "refund_issued"
    big = r.handle(TaskRequest("please refund my last order", {"amount": 400}))
    assert big.harness == "system2"
    assert any(e.kind == "blocked" for e in big.events)


def test_system2_asks_before_side_effect_then_completes():
    support_desk.CREDITS.clear()
    r = build()
    first = r.handle(TaskRequest("why did my invoice double last month?", {"account": "acme"}))
    assert first.status == "needs_user" and first.output["pending"]["tool"] == "issue_credit"
    assert support_desk.CREDITS == []
    second = r.handle(TaskRequest("why did my invoice double last month?", {"account": "acme"}),
                      approved={"issue_credit"})
    assert second.status == "completed" and "credited 49" in second.output["answer"]
    assert support_desk.CREDITS[0]["amount"] == 49


def test_system2_asks_for_missing_account():
    r = build()
    res = r.handle(TaskRequest("my bill looks wrong can you figure out why"))
    assert res.status == "needs_user" and "account" in res.output["question"]


def test_router_blocks_unlisted_tool_and_budget_stops_loop():
    calls = []
    tools = [Tool("ping", lambda: calls.append(1) or {"n": len(calls)}, max_calls=100)]
    rogue = ScriptedPlanner(lambda g, s: Step("tool", "rm_rf", {}))
    res = AgentHarness(rogue, tools, max_steps=3).run(TaskRequest("go"))
    assert res.failure.code == "budget_exhausted"
    assert sum(e.kind == "blocked" for e in res.events) == 3 and calls == []


def test_quota_and_no_progress_detection():
    tools = [Tool("same", lambda: {"v": 1}, max_calls=10)]
    looper = ScriptedPlanner(lambda g, s: Step("tool", "same", {}))
    res = AgentHarness(looper, tools, max_steps=10, max_stalls=2).run(TaskRequest("loop"))
    assert res.failure.code == "no_progress"

    tools = [Tool("counter", lambda: {"v": object()}, max_calls=2)]
    res = AgentHarness(ScriptedPlanner(lambda g, s: Step("tool", "counter", {})), tools,
                       max_steps=4, max_stalls=99).run(TaskRequest("q"))
    assert any("quota" in e.data.get("reason", "") for e in res.events)


def test_tool_exception_is_returned_as_data():
    def boom():
        raise RuntimeError("db down")
    plan = iter([Step("tool", "boom"), Step("final", message="reported outage")])
    res = AgentHarness(ScriptedPlanner(lambda g, s: next(plan)), [Tool("boom", boom)]).run(TaskRequest("x"))
    assert res.status == "completed"
    assert "db down" in res.output["state"]["facts"]["boom"]["error"]
