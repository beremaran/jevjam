"""Tests for the sleep-on-idle wrapper.

No GPU, no checkpoint download and no request to the Hub: the request path runs
against a stand-in router, and the one test that builds a real `Router` only reads
its configuration. `import laya` does not pull torch, so none of this needs CUDA.
"""
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jevjam import (
    DEFAULT_IDLE_TIMEOUT,
    RENAMED_ENV,
    IdleUnloader,
    apply_env,
    build_app,
    build_router,
    read_idle_timeout,
    warn_dead_env,
)

# Every one of these has a default, so no test may inherit the developer's shell.
MANAGED_ENV = (
    *("%s_%s" % (prefix, key) for prefix in ("JEVJAM", "LAYA") for key in RENAMED_ENV),
    "LAYA_PRELOAD",
    "LAYA_MODELS",
    "JEVJAM_MAX_LOADED",
    "LAYA_MAX_LOADED",
    "JEVJAM_CLEF_QUANT",
    "LAYA_AUTO_TASK",
    "JULIA_CPU_THREADS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)


def wait_for(predicate, timeout=5.0):
    """Poll until `predicate` holds. Returns whether it ever did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class StubRouter:
    """Stands in for laya's Router: records what the wrapper does to it.

    Duck-typed, because that is all the wrapper and `laya.serve.create_app` use.
    `predict` dispatches the lifecycle hooks the way `Router.predict` does, because
    the wrapper's whole clock hangs off them.
    """

    def __init__(self, resident=(), fail=False):
        self._resident = list(resident)
        self._fail = fail
        self.loads = []
        self.unloads = 0
        self.hooks = []

    @property
    def loaded(self):
        return list(self._resident)

    def add_hook(self, hook):
        self.hooks.append(hook)
        return self

    def load(self, name):
        self.loads.append(name)
        if name not in self._resident:
            self._resident.append(name)

    def unload(self, name=None):
        self.unloads += 1
        if name is None:
            self._resident.clear()
        else:
            self._resident.remove(name)

    def route(self, state, questions, model=None):
        """Routing only. The real Router decides here and loads nothing, and the
        tools promise the same, so a stub that loaded would be testing the wrong thing."""
        return {"model": model or "english", "repo": None, "reason": "stub"}

    def predict(self, state, questions, model=None):
        name = model or "english"
        ctx = SimpleNamespace(router=self, decision={"model": name}, model=None)
        self._dispatch("on_route", ctx)   # the last hook before Router.predict loads
        self.load(name)
        ctx.model = name
        self._dispatch("on_predict_start", ctx)
        try:
            if self._fail:
                raise RuntimeError("inference blew up")
            return {"answers": {}}
        finally:
            self._dispatch("on_predict_end", ctx)

    def _dispatch(self, event, ctx):
        for hook in self.hooks:
            method = getattr(hook, event, None)
            if method is not None:
                method(ctx)


# --------------------------------------------------------------------- configuration
def test_defaults_when_nothing_is_set():
    assert read_idle_timeout() == DEFAULT_IDLE_TIMEOUT == 300


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_value_is_the_default_not_an_error(blank, monkeypatch):
    # Compose sends an unset passthrough as an empty string; that must not stop a boot.
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", blank)
    assert read_idle_timeout() == 300


@pytest.mark.parametrize("raw, seconds", [("0", 0), ("1", 1), ("600", 600), (" 600 ", 600)])
def test_idle_timeout_is_whole_seconds(raw, seconds, monkeypatch):
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", raw)
    assert read_idle_timeout() == seconds


@pytest.mark.parametrize("raw", ["-1", "1.5", "abc", "300s", "0x10", "1,5"])
def test_a_bad_idle_timeout_stops_the_server(raw, monkeypatch):
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", raw)
    with pytest.raises(SystemExit) as exit_info:
        read_idle_timeout()
    assert "JEVJAM_IDLE_TIMEOUT" in str(exit_info.value)
    assert raw in str(exit_info.value)


def test_zero_idle_timeout_means_never_sleep(monkeypatch):
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", "0")
    assert read_idle_timeout() == 0


def test_env_that_no_longer_does_anything_is_called_out(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="jevjam"):
        warn_dead_env()
    assert caplog.records == []

    monkeypatch.setenv("LAYA_PRELOAD", "1")
    monkeypatch.setenv("LAYA_MODELS", "english")
    monkeypatch.setenv("JEVJAM_MAX_LOADED", "2")
    with caplog.at_level(logging.WARNING, logger="jevjam"):
        warn_dead_env()
    assert len(caplog.records) == 1
    assert "LAYA_PRELOAD" in caplog.records[0].getMessage()
    assert "LAYA_MODELS" in caplog.records[0].getMessage()
    assert "JEVJAM_MAX_LOADED" in caplog.records[0].getMessage()


def test_a_blank_dead_variable_is_not_worth_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("LAYA_MODELS", "")  # what an unset Compose passthrough looks like
    with caplog.at_level(logging.WARNING, logger="jevjam"):
        warn_dead_env()
    assert caplog.records == []


# -------------------------------------------------------------------------- watcher
def test_every_checkpoint_is_freed_after_the_idle_window():
    router = StubRouter(resident=["english", "multilingual"])
    idle = IdleUnloader(router, timeout=1)
    idle.start()
    try:
        idle.touch()  # the request that loaded them, and the clock the deadline runs on
        time.sleep(0.3)
        assert router.unloads == 0, "unloaded before the window was up"
        assert wait_for(lambda: router.unloads == 1)
        assert router.loaded == []
    finally:
        idle.stop()


def test_a_request_moves_the_deadline():
    router = StubRouter(resident=["english"])
    idle = IdleUnloader(router, timeout=1)
    idle.start()
    try:
        time.sleep(0.7)
        idle.touch()
        time.sleep(0.7)  # 1.4s from the start, but only 0.7s from the request
        assert router.unloads == 0, "the deadline did not move"
        assert wait_for(lambda: router.unloads == 1)
    finally:
        idle.stop()


def test_the_watcher_waits_for_the_first_request():
    router = StubRouter()  # nothing resident, so nothing to free
    idle = IdleUnloader(router, timeout=1)
    idle.start()
    try:
        assert idle.last_activity is None
        time.sleep(1.3)
        assert router.unloads == 0
    finally:
        idle.stop()


def test_timeout_zero_keeps_the_checkpoints():
    router = StubRouter(resident=["english"])
    idle = IdleUnloader(router, timeout=0)
    idle.start()
    idle.touch()
    time.sleep(0.3)
    assert router.unloads == 0
    assert router.loaded == ["english"]


def test_stopping_the_server_stops_the_watcher():
    router = StubRouter(resident=["english"])
    idle = IdleUnloader(router, timeout=1)
    idle.start()
    idle.touch()
    idle.stop()
    time.sleep(1.3)
    assert router.unloads == 0


# --------------------------------------------------------------------- request path
def test_a_request_reaches_the_router_and_stamps_the_clock():
    router = StubRouter()
    app, built_router, idle = build_app(router=router, timeout=60, others=[])
    assert built_router.default is router, "the app and the watcher must share one router"
    assert len(router.hooks) == 1, "the wrapper installs exactly one hook"

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "loaded": [], "device": "auto", "mcp_ready": True,
        }
        assert idle.last_activity is None, "a health probe must not keep the checkpoints"

        answer = client.post("/v1/systemone", json={
            "state": "I was charged twice",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        })
        assert answer.status_code == 200
        assert router.loads == ["english"]
        assert idle.last_activity is not None
        assert client.get("/health").json()["loaded"] == ["english"]


def test_models_lists_every_backend_and_honours_the_api_key(monkeypatch):
    monkeypatch.setenv("JEVJAM_API_KEY", "secret")
    app, _, _ = build_app(router=StubRouter(), timeout=60)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        listing = client.get("/v1/models", headers={"Authorization": "Bearer secret"}).json()
    assert listing["object"] == "list"
    assert [m["id"] for m in listing["data"]] == [
        "auto", "english", "multilingual", "typed-decisions", "julia-1", "clef-flash"]


def test_a_request_that_fails_still_counts_as_activity():
    # Traffic is traffic: a client whose inference keeps failing should not be the
    # reason the server holds GPU memory all day.
    router = StubRouter(fail=True)
    app, _router, idle = build_app(router=router, timeout=60)
    with TestClient(app) as client:
        answer = client.post("/v1/systemone", json={
            "state": "I was charged twice",
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        })
        assert answer.status_code == 500
        assert idle.last_activity is not None


def test_the_router_holds_one_checkpoint(monkeypatch):
    monkeypatch.setenv("LAYA_DEVICE", "cpu")
    monkeypatch.setenv("LAYA_AUTO_TASK", "1")
    router = build_router()
    assert router.max_loaded == 1
    assert router.auto_task_detection is True
    assert router.loaded == [], "a cold server holds nothing"
