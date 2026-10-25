import pandas as pd

from supportpilot.classifier.data import fingerprint, split
from supportpilot.classifier.predict import URGENT


def _frame(n: int = 200) -> pd.DataFrame:
    queues = ["Billing", "Technical", "Sales", "HR"]
    return pd.DataFrame(
        {
            "ticket_id": range(n),
            "text": [f"ticket {i}" for i in range(n)],
            "queue": [queues[i % 4] for i in range(n)],
            "priority": ["low"] * n,
        }
    )


def test_split_is_deterministic_disjoint_and_stratified():
    df = _frame()
    a, b = split(df), split(df)
    assert [fingerprint(x) for x in a] == [fingerprint(x) for x in b]
    train, val, test = a
    assert len(train) == 160 and len(val) == 20 and len(test) == 20
    ids = [set(x["ticket_id"]) for x in a]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert test["queue"].value_counts().to_dict() == {"Billing": 5, "Technical": 5, "Sales": 5, "HR": 5}


def test_urgent_rule():
    assert URGENT.search("We suspect a data breach in our tenant")
    assert URGENT.search("The platform is down for every customer")
    assert URGENT.search("complete outage since 9am")
    assert not URGENT.search("How do I reset my password?")
    assert not URGENT.search("Search results are slow to load")
