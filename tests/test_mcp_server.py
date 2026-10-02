"""Tests for the MCP endpoint.

Same ground rules as the HTTP tests: no GPU, no checkpoint download, no call to the
Hub. Most tests call the registered tool functions directly; the integration test
also sends a streamable-HTTP tools/call request through `/mcp`. `import
laya.mcp.server` pulls no torch, so none of this needs CUDA.
"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from laya.mcp import server as laya_mcp
from test_server import MANAGED_ENV, StubRouter, wait_for

from jevjam import DEFAULT_IDLE_TIMEOUT, IdleUnloader, build_app

QUESTIONS = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
STATE = {"body": "I was charged twice"}


def call_mcp_predict(client, authorization=None):
    headers = {"accept": "application/json"}
    if authorization is not None:
        headers["authorization"] = authorization
    return client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "jevjam_predict",
                "arguments": {"state": STATE, "questions": QUESTIONS},
            },
        },
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    # The router lives in a module global, so a test that sets it would hand the next
    # test a router with a watcher attached to a finished test's stub.
    monkeypatch.setattr(laya_mcp, "_ROUTER", None, raising=False)


# ------------------------------------------------------------------------ tool layer
def test_the_four_tools_are_registered_under_jevjam_names():
    build_app(router=StubRouter(), timeout=60)
    names = {tool.name for tool in asyncio.run(laya_mcp.server.list_tools())}
    assert names == {"jevjam_predict", "jevjam_route", "jevjam_preset", "jevjam_status"}


def test_a_predict_call_reaches_our_router_and_stamps_the_clock():
    router = StubRouter()
    _app, built, idle = build_app(router=router, timeout=60)
    assert built.default is router, "the tools and the watcher must share one router"
    assert len(router.hooks) == 1, "the wrapper installs exactly one hook"

    assert idle.last_activity is None
    answer = laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS)

    assert '"answers"' in answer, "the tool returns its payload as JSON text"
    assert router.loads == ["english"]
    assert idle.last_activity is not None, "an MCP call must keep the checkpoints as long as an HTTP one"
    assert router.loaded == ["english"]


def test_a_preset_call_stamps_the_clock_too():
    # laya_preset runs its forward pass through laya_predict, so it is the second
    # door into Router.predict and the one most likely to be forgotten.
    router = StubRouter()
    _app, _built, idle = build_app(router=router, timeout=60)
    laya_mcp.laya_preset_tool(preset="triage", state=STATE)
    assert router.loads == ["english"]
    assert idle.last_activity is not None


def test_a_failed_call_still_counts_as_activity():
    router = StubRouter(fail=True)
    _app, _built, idle = build_app(router=router, timeout=60)
    with pytest.raises(laya_mcp.McpToolError):
        laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS)
    assert idle.last_activity is not None, "traffic is traffic, even when it fails"


def test_status_reports_the_shared_router_and_a_cold_start():
    router = StubRouter()
    _app, _built, _idle = build_app(router=router, timeout=60)
    status = laya_mcp.laya_status_tool()
    assert '"router_ready": true' in status
    assert '"loaded": []' in status, "nothing is resident until a tool asks for something"


def test_a_route_call_alone_does_not_pull_in_a_checkpoint():
    router = StubRouter()
    _app, _built, _idle = build_app(router=router, timeout=60)
    laya_mcp.laya_route_tool(state=STATE, questions=QUESTIONS)
    assert router.loads == [], "laya_route promises no forward pass"


def test_the_idle_timeout_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("JEVJAM_IDLE_TIMEOUT", "7")
    _app, _router, idle = build_app(router=StubRouter())
    assert idle._timeout == 7, "the MCP server obeys the same clock as the HTTP one"

    monkeypatch.delenv("JEVJAM_IDLE_TIMEOUT")
    assert build_app(router=StubRouter())[2]._timeout == float(DEFAULT_IDLE_TIMEOUT)


# -------------------------------------------------------------------------- watching
def test_the_checkpoints_are_freed_after_the_idle_window():
    router = StubRouter()
    _app, _built, idle = build_app(router=router, timeout=1)
    assert isinstance(idle, IdleUnloader)
    idle.start()
    try:
        laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS)
        assert router.loaded == ["english"]
        assert wait_for(lambda: router.unloads == 1)
        assert router.loaded == []
    finally:
        idle.stop()


def test_a_cold_server_holds_nothing_before_the_first_call():
    router = StubRouter()
    _app, _built, _idle = build_app(router=router, timeout=1)
    assert router.loaded == [], "no preload, same as the HTTP server"


def test_the_api_and_mcp_tools_share_one_router_and_idle_clock():
    router = StubRouter()
    app, built, idle = build_app(router=router, timeout=60)
    assert built.default is router
    assert laya_mcp._ROUTER is built

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "loaded": [],
            "device": "auto",
            "mcp_ready": True,
        }
        api_response = client.post(
            "/v1/systemone",
            json={"state": STATE, "questions": QUESTIONS},
        )
        assert api_response.status_code == 200, api_response.text
        assert router.loads == ["english"]
        assert idle.last_activity is not None

        response = call_mcp_predict(client)

        assert response.status_code == 200, response.text
        result_text = response.json()["result"]["content"][0]["text"]
        assert "answers" in json.loads(result_text)
        assert router.loads == ["english", "english"]
        assert idle.last_activity is not None


@pytest.mark.parametrize("authorization", [None, "Basic secret", "Bearer", "Bearer wrong"])
def test_configured_key_rejects_missing_or_invalid_mcp_bearer(authorization, monkeypatch):
    monkeypatch.setenv("JEVJAM_API_KEY", "secret")
    router = StubRouter()
    app, _built, idle = build_app(router=router, timeout=60)

    with TestClient(app) as client:
        response = call_mcp_predict(client, authorization)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert router.loads == [], "rejected MCP calls must not reach the router"
    assert idle.last_activity is None, "rejected calls must not reset the idle clock"


def test_a_configured_key_protects_every_mcp_method_and_leaves_health_public(monkeypatch):
    monkeypatch.setenv("JEVJAM_API_KEY", "secret")
    app, _built, _idle = build_app(router=StubRouter(), timeout=60)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/mcp").status_code == 401


def test_the_http_api_and_mcp_share_the_configured_bearer_key(monkeypatch):
    monkeypatch.setenv("JEVJAM_API_KEY", "secret")
    router = StubRouter()
    app, _built, _idle = build_app(router=router, timeout=60)

    with TestClient(app) as client:
        api_request = {"state": STATE, "questions": QUESTIONS}
        assert client.post("/v1/systemone", json=api_request).status_code == 401
        assert client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer secret"},
            json=api_request,
        ).status_code == 200

        assert call_mcp_predict(client).status_code == 401
        response = call_mcp_predict(client, "Bearer secret")
        assert response.status_code == 200, response.text
        assert router.loads == ["english", "english"]


def test_a_missing_shared_router_fails_instead_of_building_another():
    router = StubRouter()
    build_app(router=router, timeout=60)
    laya_mcp._ROUTER = None

    with pytest.raises(laya_mcp.McpToolError, match="shared router is not configured"):
        laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS)

    assert router.loads == []
