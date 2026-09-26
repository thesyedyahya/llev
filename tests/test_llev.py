import math

import pytest
from fastapi.testclient import TestClient

from llev.api import build_service, create_app
from llev.config import Settings
from llev.engine import FakeEngine
from llev.prompts import LETTERS, _orders
from llev.schemas import ChoiceQ
from llev.prompts import choice_jobs
from llev.templates import get_template

KEY = "test-key"
H = {"Authorization": f"Bearer {KEY}"}


def settings(tmp_path, **kw) -> Settings:
    # _env_file=None: tests must not pick up a developer's local .env
    return Settings(_env_file=None, engine="fake", model="fake", api_keys=KEY, data_dir=str(tmp_path), **kw)


def client(tmp_path, primary=None, escalation=None, **kw) -> TestClient:
    s = settings(tmp_path, **kw)
    return TestClient(create_app(s, build_service(s, primary or FakeEngine(), escalation)))


TICKET = {
    "state": {"message": "I was charged twice for my order, please refund the payment"},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team handles this",
                 "criteria": {"billing": "payment refund charged", "tech": "app crash bug",
                              "safety": "danger harassment"}},
        "anger": {"type": "score", "instructions": "How upset", "criteria": ["calm", "annoyed", "furious"]},
        "refund": {"type": "noul", "instructions": "asks for a refund"},
        "tags": {"type": "multi", "instructions": "Topics", "criteria": {"pay": "payment charged",
                                                                          "bug": "crash freeze"}},
    },
}


def test_orders_are_distinct_rotations():
    assert _orders(4, 2) == [[0, 1, 2, 3], [2, 3, 0, 1]]
    assert _orders(3, 5) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for order in _orders(7, 3):
        assert sorted(order) == list(range(7))


def test_choice_jobs_map_fewshot_labels_through_permutation():
    q = ChoiceQ(instructions="x", criteria={"a": "1", "b": "2", "c": "3", "d": "4"})
    jobs = choice_jobs(get_template("chatml"), q, 2, [("st", "a")])
    # second ordering starts at option index 2, so "a" (index 0) is shown as letter C
    assert "Example 1 answer: A" in jobs[0].suffix
    assert "Example 1 answer: C" in jobs[1].suffix
    assert jobs[1].suffix.endswith("<|im_start|>assistant\nAnswer:")


def test_decide_all_types(tmp_path):
    r = client(tmp_path).post("/v1/decide", json=TICKET, headers=H)
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["team"]["choice"] == "billing"
    assert math.isclose(sum(a["team"]["probabilities"].values()), 1, abs_tol=1e-4)
    assert 0 <= a["anger"]["score"] <= 2 and len(a["anger"]["probabilities"]) == 3
    assert a["refund"]["noul"] > 0.5
    assert a["tags"]["selected"] == ["pay"]
    assert all(0 <= x["confidence"] <= 1 for x in a.values())


def test_jev_compatible_alias(tmp_path):
    assert client(tmp_path).post("/v1/systemone", json=TICKET, headers=H).status_code == 200


def test_auth_required(tmp_path):
    c = client(tmp_path)
    assert c.post("/v1/decide", json=TICKET).status_code == 401
    assert c.post("/v1/decide", json=TICKET, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/v1/decide", json=TICKET, headers={"x-api-key": KEY}).status_code == 200


def test_refuses_to_start_without_keys(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(Settings(_env_file=None, engine="fake", model="fake", data_dir=str(tmp_path)))


def test_position_bias_is_averaged_out(tmp_path):
    # A model that always prefers the first letter shown: debiasing should spread its vote.
    def biased(prefix, suffix, labels):
        return {lbl: (0.0 if lbl == "A" else -5.0) for lbl in labels}

    q = {"state": "x", "questions": {"c": {"type": "choice", "instructions": "pick",
                                           "criteria": {"p": "1", "q": "2"}}}}
    one = client(tmp_path / "a", FakeEngine(scorer=biased), debias_permutations=1)
    two = client(tmp_path / "b", FakeEngine(scorer=biased), debias_permutations=2)
    p1 = one.post("/v1/decide", json=q, headers=H).json()["answers"]["c"]
    p2 = two.post("/v1/decide", json=q, headers=H).json()["answers"]["c"]
    assert p1["confidence"] > 0.99
    assert math.isclose(p2["probabilities"]["p"], 0.5, abs_tol=1e-6)


def test_low_confidence_escalates(tmp_path):
    unsure = FakeEngine("small", scorer=lambda p, s, labels: {l: 0.0 for l in labels})
    sure = FakeEngine("big")
    c = client(tmp_path, unsure, sure, escalate_threshold=0.6)
    a = c.post("/v1/decide", json=TICKET, headers=H).json()["answers"]
    assert a["team"]["escalated"] and a["team"]["engine"] == "big"
    assert a["team"]["choice"] == "billing"

    no_esc = dict(TICKET, escalate=False)
    a = c.post("/v1/decide", json=no_esc, headers=H).json()["answers"]
    assert a["team"]["engine"] == "small" and not a["team"].get("escalated")


def test_feedback_becomes_fewshot_memory(tmp_path):
    seen = []

    def spy(prefix, suffix, labels):
        seen.append(suffix)
        return {l: 0.0 for l in labels}

    c = client(tmp_path, FakeEngine(scorer=spy), fewshot_k=2)
    q = {"state": "customer left phone in the shop", "questions": {
        "cat": {"type": "choice", "task": "ticket.cat", "instructions": "category",
                "criteria": {"lost": "lost item", "pay": "payment"}}}}
    d = c.post("/v1/decide", json=q, headers=H).json()
    assert d["answers"]["cat"]["fewshot"] == 0

    fb = c.post("/v1/feedback", json={"id": d["id"], "key": "cat", "label": "lost"}, headers=H)
    assert fb.json() == {"ok": True, "learned": True, "detail": None}
    assert c.get("/v1/memory", headers=H).json() == {"tasks": {"ticket.cat": 1}}

    # default knn mode: the similar corrected case votes; the prompt stays clean
    q["state"] = "customer forgot phone in the shop"
    seen.clear()
    d2 = c.post("/v1/decide", json=q, headers=H).json()["answers"]["cat"]
    assert d2["fewshot"] == 1
    assert d2["choice"] == "lost" and d2["probabilities"]["lost"] > 0.6
    assert "customer left phone in the shop" not in seen[0]

    # memory survives restart; prompt mode shows the example to the model instead
    seen.clear()
    c2 = client(tmp_path, FakeEngine(scorer=spy), fewshot_k=2, fewshot_mode="prompt")
    assert c2.get("/v1/memory", headers=H).json() == {"tasks": {"ticket.cat": 1}}
    d3 = c2.post("/v1/decide", json=q, headers=H).json()["answers"]["cat"]
    assert "customer left phone in the shop" in seen[0]
    assert math.isclose(d3["probabilities"]["lost"], 0.5, abs_tol=1e-6)


@pytest.mark.parametrize("label", ["nope", 3, None])
def test_feedback_validates_label(tmp_path, label):
    c = client(tmp_path)
    d = c.post("/v1/decide", json=TICKET, headers=H).json()
    assert c.post("/v1/feedback", json={"id": d["id"], "key": "team", "label": label},
                  headers=H).status_code == 422


def test_feedback_unknown_id(tmp_path):
    r = client(tmp_path).post("/v1/feedback", json={"id": "dec_x", "key": "team", "label": "billing"},
                              headers=H)
    assert r.status_code == 404


def test_validation_limits(tmp_path):
    c = client(tmp_path)
    too_many = {"state": "x", "questions": {"c": {"type": "choice", "instructions": "i",
                                                  "criteria": {f"o{i}": "d" for i in range(27)}}}}
    assert c.post("/v1/decide", json=too_many, headers=H).status_code == 422
    big = {"state": "x" * 70_000, "questions": {"n": {"type": "noul", "instructions": "i"}}}
    assert c.post("/v1/decide", json=big, headers=H).status_code == 413
    assert LETTERS[25] == "Z"


def test_playground_served_without_key(tmp_path):
    r = client(tmp_path).get("/playground")
    assert r.status_code == 200 and "LLEV Playground" in r.text


def test_accurate_tier_skips_small_model(tmp_path):
    small, big = FakeEngine("small"), FakeEngine("big")
    c = client(tmp_path, small, big)
    q = {"state": "I was charged twice, refund the payment", "questions": {
        "danger": {"type": "noul", "tier": "accurate", "instructions": "someone is in danger"},
        "refund": {"type": "noul", "instructions": "asks for a refund payment"}}}
    a = c.post("/v1/decide", json=q, headers=H).json()["answers"]
    assert list(a) == ["danger", "refund"]
    assert a["danger"]["engine"] == "big" and not a["danger"].get("escalated")
    assert a["refund"]["engine"] == "small"
    assert small.calls == 1 and big.calls == 1

    # no escalation model (or escalate=false): accurate falls back to the primary
    a = c.post("/v1/decide", json=dict(q, escalate=False), headers=H).json()["answers"]
    assert a["danger"]["engine"] == "small"
