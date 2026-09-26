"""Tests for the MCP server.

Same ground rules as the HTTP tests: no GPU, no checkpoint download, no call to the
Hub. The tools are called directly rather than over a socket, because
`laya.mcp.server` registers them with `@server.tool` and the decorator hands the
function back unchanged, so `laya_predict_tool(...)` is the tool the MCP client would
reach. `import laya.mcp.server` pulls no torch, so none of this needs CUDA.
"""
import asyncio

import pytest
from laya.mcp import server as laya_mcp
from test_idle_serve import MANAGED_ENV, StubRouter, wait_for

from laya_idle_serve import DEFAULT_IDLE_TIMEOUT, IdleUnloader
from laya_idle_serve.mcp_server import build_mcp_server

QUESTIONS = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}
STATE = {"body": "I was charged twice"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    # The router lives in a module global, so a test that sets it would hand the next
    # test a router with a watcher attached to a finished test's stub.
    monkeypatch.setattr(laya_mcp, "_ROUTER", None, raising=False)


# ------------------------------------------------------------------------ tool layer
def test_the_four_tools_are_registered():
    names = {tool.name for tool in asyncio.run(laya_mcp.server.list_tools())}
    assert names == {"laya_predict", "laya_route", "laya_preset", "laya_status"}


def test_a_predict_call_reaches_our_router_and_stamps_the_clock():
    router = StubRouter()
    _server, built, idle = build_mcp_server(router=router, timeout=60, max_loaded=1)
    assert built is router, "the tools and the watcher must share one router"
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
    _server, _built, idle = build_mcp_server(router=router, timeout=60)
    laya_mcp.laya_preset_tool(preset="triage", state=STATE)
    assert router.loads == ["english"]
    assert idle.last_activity is not None


def test_a_failed_call_still_counts_as_activity():
    router = StubRouter(fail=True)
    _server, _built, idle = build_mcp_server(router=router, timeout=60)
    with pytest.raises(laya_mcp.McpToolError):
        laya_mcp.laya_predict_tool(state=STATE, questions=QUESTIONS)
    assert idle.last_activity is not None, "traffic is traffic, even when it fails"


def test_status_reports_the_shared_router_and_a_cold_start():
    router = StubRouter()
    _server, _built, _idle = build_mcp_server(router=router, timeout=60)
    status = laya_mcp.laya_status_tool()
    assert '"router_ready": true' in status
    assert '"loaded": []' in status, "nothing is resident until a tool asks for something"


def test_a_route_call_alone_does_not_pull_in_a_checkpoint():
    router = StubRouter()
    build_mcp_server(router=router, timeout=60)
    laya_mcp.laya_route_tool(state=STATE, questions=QUESTIONS)
    assert router.loads == [], "laya_route promises no forward pass"


def test_the_idle_timeout_and_cap_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LAYA_IDLE_TIMEOUT", "7")
    monkeypatch.setenv("LAYA_MAX_LOADED", "1")
    _server, _router, idle = build_mcp_server(router=StubRouter())
    assert idle._timeout == 7, "the MCP server obeys the same clock as the HTTP one"

    monkeypatch.delenv("LAYA_IDLE_TIMEOUT")
    assert build_mcp_server(router=StubRouter())[2]._timeout == float(DEFAULT_IDLE_TIMEOUT)


# -------------------------------------------------------------------------- watching
def test_the_checkpoints_are_freed_after_the_idle_window():
    router = StubRouter()
    _server, _built, idle = build_mcp_server(router=router, timeout=1)
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
    _server, _built, _idle = build_mcp_server(router=router, timeout=1)
    assert router.loaded == [], "no preload, same as the HTTP server"
