"""A billing support desk wired with both harnesses.

System 1 labels:   password_reset, refund, plan_question, investigate
System 2 tools:    get_invoices, get_plan_changes, issue_credit (approval required)
"""
from __future__ import annotations

from harness import (AgentHarness, AppRouter, DecisionHarness, NaiveBayesDecider,
                     ScriptedPlanner, Step, Tool)

EXAMPLES = {
    "password_reset": [
        "I forgot my password", "reset my password please", "can't log in, need a password reset",
        "locked out of my account password", "how do I change my login password",
    ],
    "refund": [
        "refund order 1234", "I want my money back for order 88", "please refund my last order",
        "request a refund for the duplicate order", "refund the order I cancelled",
    ],
    "plan_question": [
        "what does the pro plan include", "how much is the team plan", "which plan has sso",
        "compare the basic and pro plans", "does the pro plan include api access",
    ],
    "investigate": [
        "why did my invoice double last month", "my bill looks wrong can you figure out why",
        "charges changed and I don't understand why", "something is off with my invoices please look into it",
        "investigate why I was charged twice this month",
    ],
}

# ------------------------------------------------------------------ fake backend
INVOICES = {"acme": [{"month": "2026-07", "amount": 49}, {"month": "2026-08", "amount": 98}]}
PLAN_CHANGES = {"acme": [{"date": "2026-08-01", "from": "basic", "to": "pro", "seats": 1},
                         {"date": "2026-08-03", "from": "pro", "to": "basic", "seats": 1}]}
CREDITS: list[dict] = []


def get_invoices(account: str):
    return INVOICES.get(account, [])


def get_plan_changes(account: str):
    return PLAN_CHANGES.get(account, [])


def issue_credit(account: str, amount: int, reason: str):
    CREDITS.append({"account": account, "amount": amount, "reason": reason})
    return {"credited": amount}


TOOLS = [
    Tool("get_invoices", get_invoices, "List monthly invoices for an account"),
    Tool("get_plan_changes", get_plan_changes, "List plan changes for an account"),
    Tool("issue_credit", issue_credit, "Credit an account (side effect)", requires_approval=True, max_calls=1),
]


# ------------------------------------------------------------- scripted planner
def billing_policy(goal: str, state: dict) -> Step:
    """What an LLM planner would plausibly do, written as a deterministic function."""
    facts = state["facts"]
    account = state.get("account")
    if not account:
        return Step("ask", message="Which account is this about?", thought="need the account id")
    if "get_invoices" not in facts:
        return Step("tool", "get_invoices", {"account": account}, thought="look at the bills first")
    inv = facts["get_invoices"]
    if len(inv) >= 2 and inv[-1]["amount"] > inv[-2]["amount"] and "get_plan_changes" not in facts:
        return Step("tool", "get_plan_changes", {"account": account}, thought="amount jumped; check plan changes")
    changes = facts.get("get_plan_changes", [])
    reverted = len(changes) >= 2 and changes[0]["from"] == changes[-1]["to"]
    if reverted and "issue_credit" not in facts:
        extra = inv[-1]["amount"] - inv[-2]["amount"]
        return Step("tool", "issue_credit",
                    {"account": account, "amount": extra, "reason": "upgrade reverted within 3 days"},
                    thought="upgrade was reverted; the difference should be credited")
    if "issue_credit" in facts:
        return Step("final", message=f"Your August invoice included a pro upgrade you reverted two days later. "
                                     f"I credited {facts['issue_credit']['credited']} to your account.")
    return Step("final", message="Your invoices look consistent with your plan history.")


def build(threshold: float = 0.6, refund_limit: int = 100) -> AppRouter:
    def refund_policy(req, d):
        amount = req.state.get("amount", 0)
        return f"refund {amount} exceeds auto-approve limit {refund_limit}" if amount > refund_limit else None

    fast = DecisionHarness(
        NaiveBayesDecider(EXAMPLES),
        actions={
            "password_reset": lambda r, d: {"action": "sent_reset_link"},
            "refund": lambda r, d: {"action": "refund_issued", "amount": r.state.get("amount", 0)},
            "plan_question": lambda r, d: {"action": "answered_from_faq"},
        },
        threshold=threshold,
        policies={"refund": refund_policy},
        escalate_labels=("investigate",),
    )
    slow = AgentHarness(
        ScriptedPlanner(billing_policy), TOOLS,
        constraints=["never credit more than the disputed difference", "ask before side effects"],
        max_steps=6,
    )
    return AppRouter(fast, slow)
