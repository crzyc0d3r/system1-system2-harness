"""Run a few support requests through System 1 -> System 2 and print the event stream.

Run:  python demo.py
"""
from harness import TaskRequest
from examples.support_desk import build, CREDITS

router = build()

CASES = [
    ("reset my password please", {}, None),
    ("please refund my last order", {"amount": 40}, None),
    ("please refund my last order", {"amount": 400}, None),          # policy gate -> escalate
    ("why did my invoice double last month?", {"account": "acme"}, None),  # stops to ask approval
    ("why did my invoice double last month?", {"account": "acme"}, {"issue_credit"}),  # approved
]

for text, state, approved in CASES:
    print("=" * 78)
    print(f"REQUEST: {text!r}  state={state}  approved={approved or set()}")
    res = router.handle(TaskRequest(text, dict(state)), approved=approved)
    for e in res.events:
        print("  ", e)
    shown = {k: v for k, v in res.output.items() if k != "state"}
    print(f"-> {res.harness}: {res.status} {shown}" + (f" failure={res.failure}" if res.failure else ""))

print("=" * 78)
print("credits issued:", CREDITS)
