"""Tests for the Julia backend, the shared resident limit and the JEVJAM_* settings.

No GPU, no download: Julia's engine is a stand-in that scores with fixed logits, but
the answers go through Julia's real `predict_typed`, so its validation and output
shape are the ones the server sees in production.
"""
import logging
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from julia.typed import predict_typed
from laya.mcp import server as laya_mcp
from test_server import MANAGED_ENV, StubRouter, wait_for

from jevjam import apply_env, build_app
from jevjam.models import REVISION, Julia

STATE = {"body": "I was charged twice. Refund it or I cancel."}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "payments", "shipping": "deliveries"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "churn": {"type": "noul", "instructions": "Will they cancel?"},
}


class FakeEngine:
    """Julia's engine surface, scoring the first option highest."""

    device = "cpu"

    def logits(self, rows):
        return [[float(len(row["options"]) - i) for i in range(len(row["options"]))] for row in rows]

    def predict(self, state, questions):
        return predict_typed(self, state, questions)

    def encoding_info(self, rows):
        return [{"tokens": 10} for _ in rows]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(laya_mcp, "_ROUTER", None, raising=False)


def julia():
    return Julia(loader=FakeEngine)


def ask(client, model, questions=QUESTIONS):
    return client.post("/v1/systemone", json={"state": STATE, "questions": questions, "model": model})


# ------------------------------------------------------------------------- answers
@pytest.mark.parametrize("model", ["julia-1", "SupersonicLabs/Julia-1", " Julia "])
def test_a_request_that_names_julia_gets_a_jev_shaped_answer(model):
    router = StubRouter()
    app, models, idle = build_app(router=router, timeout=60, others=[julia()])
    with TestClient(app) as client:
        response = ask(client, model)
        assert response.status_code == 200, response.text
        body = response.json()
        assert router.loads == [], "Laya must not answer a Julia request"
        assert client.get("/health").json()["loaded"] == ["julia-1"]
    assert idle.last_activity is not None

    assert body["model"] == "julia-1"
    assert body["routing"]["repo"] == "SupersonicLabs/Julia-1"
    assert body["usage"] == {"input_tokens": 30, "output_tokens": 0}
    team, urgency, churn = (body["answers"][k] for k in ("team", "urgency", "churn"))
    assert team["choice"] == "billing"
    assert team["confidence"] == team["answer_confidence"] == max(team["probabilities"].values())
    assert set(urgency) >= {"score", "probabilities", "confidence", "answer_confidence"}
    assert churn["noul"] < 0.5, "the stand-in favours `false`"
    assert churn["answer_confidence"] == pytest.approx(1 - churn["noul"])
    assert "max_probability" not in team


def test_other_models_still_go_to_laya():
    router = StubRouter()
    app, _models, _idle = build_app(router=router, timeout=60, others=[julia()])
    with TestClient(app) as client:
        assert ask(client, "jev-1").status_code == 200
    assert router.loads == ["english"]


def test_list_criteria_work_for_julia_as_they_do_for_laya():
    app, _models, _idle = build_app(router=StubRouter(), timeout=60, others=[julia()])
    questions = {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "shipping"]}}
    with TestClient(app) as client:
        response = ask(client, "julia-1", questions)
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["team"]["choice"] == "billing"


def test_a_julia_validation_error_names_the_question():
    app, _models, _idle = build_app(router=StubRouter(), timeout=60, others=[julia()])
    questions = {**QUESTIONS, "lonely": {"type": "choice", "instructions": "Pick", "criteria": {"only": "one"}}}
    with TestClient(app) as client:
        response = ask(client, "julia-1", questions)
    assert response.status_code == 422
    assert response.json()["detail"].startswith("question 'lonely': options must contain 2")


def test_mcp_predict_reaches_julia():
    _app, models, _idle = build_app(router=StubRouter(), timeout=60, others=[julia()])
    answer = laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS, model="julia-1")
    assert '"julia-1"' in answer
    assert models.loaded == ["julia-1"]
    assert '"julia-1": "cpu"' in laya_mcp.laya_status_tool(), "status reports Julia's device"


# ------------------------------------------------------------------ resident limit
def test_laya_and_julia_share_one_resident_limit():
    router = StubRouter()
    app, models, _idle = build_app(router=router, timeout=60, max_loaded=1, others=[julia()])
    with TestClient(app) as client:
        assert ask(client, None).status_code == 200
        assert models.loaded == ["english"]
        assert ask(client, "julia-1").status_code == 200
        assert models.loaded == ["julia-1"], "loading Julia frees Laya's checkpoint"
        assert ask(client, None).status_code == 200
        assert models.loaded == ["english"], "and the other way round"


def test_the_least_recently_used_checkpoint_goes_first():
    router = StubRouter()
    app, models, _idle = build_app(router=router, timeout=60, max_loaded=2, others=[julia()])
    with TestClient(app) as client:
        ask(client, "julia-1")
        ask(client, "english")
        ask(client, "julia-1")            # julia is now the most recent
        ask(client, "multilingual")
    assert sorted(models.loaded) == ["julia-1", "multilingual"]


def test_the_idle_watcher_frees_julia_too():
    app, models, _idle = build_app(router=StubRouter(), timeout=1, others=[julia()])
    with TestClient(app) as client:  # the app's lifespan runs the watcher
        ask(client, "julia-1")
        assert models.loaded == ["julia-1"]
        assert wait_for(lambda: models.loaded == [])


def test_julia_code_and_weights_come_from_one_commit():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert re.search(r'supersonic-julia = \{ git = "[^"]+", rev = "%s" \}' % REVISION, pyproject)


# ------------------------------------------------------------------------ settings
def test_jevjam_settings_reach_laya_and_julia(monkeypatch):
    monkeypatch.setenv("JEVJAM_DEVICE", "cpu")
    monkeypatch.setenv("JEVJAM_THREADS", "3")
    apply_env()
    assert os.environ["LAYA_DEVICE"] == "cpu"
    assert os.environ["LAYA_THREADS"] == os.environ["JULIA_CPU_THREADS"] == "3"


def test_an_old_laya_name_still_works_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("LAYA_IDLE_TIMEOUT", "7")
    monkeypatch.setenv("LAYA_API_KEY", "secret")
    with caplog.at_level(logging.WARNING, logger="jevjam"):
        _app, _models, idle = build_app(router=StubRouter(), others=[julia()])
    assert idle._timeout == 7
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "LAYA_IDLE_TIMEOUT is deprecated; use JEVJAM_IDLE_TIMEOUT" in messages
    assert "LAYA_API_KEY is deprecated" in messages


def test_the_new_name_wins_over_the_old_one(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "old")
    monkeypatch.setenv("JEVJAM_API_KEY", "new")
    app, _models, _idle = build_app(router=StubRouter(), timeout=60, others=[julia()])
    with TestClient(app) as client:
        request = {"state": STATE, "questions": QUESTIONS}
        assert client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer old"}).status_code == 401
        assert client.post("/v1/systemone", json=request, headers={"Authorization": "Bearer new"}).status_code == 200


def test_a_blank_new_name_is_unset(monkeypatch):
    # Compose passes an unset passthrough as an empty string.
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", "")
    monkeypatch.setenv("LAYA_IDLE_TIMEOUT", "9")
    assert build_app(router=StubRouter(), others=[julia()])[2]._timeout == 9
