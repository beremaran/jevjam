"""Laya as an MCP server, with the checkpoints freed when it goes idle.

laya ships an MCP server of its own (`laya-mcp-server`), but it does not fit this
image twice over. It speaks stdio, so the MCP client owns the process, which is no
use to a server that is meant to keep running; and it holds every checkpoint it has
loaded for the life of the process, which is the one thing this image exists to
stop doing.

So this serves the same four tools (`laya_predict`, `laya_route`, `laya_preset`,
`laya_status`) over streamable HTTP, and borrows the sleep-on-idle wrapper
unchanged: same router, same `LAYA_IDLE_TIMEOUT`, same `LAYA_MAX_LOADED`. The idle
clock comes for free because the two tools that run a forward pass
(`laya_predict`, and `laya_preset` through it) both reach `Router.predict`, which is
where the wrapper's hooks are stamped.

What is borrowed from laya is the tool layer and nothing else. `laya.mcp.server`
keeps the router in a module global and hands that global to every tool, so
replacing it is what points the tools at the idle-aware router. The
`laya[serve,mcp]==0.3.20` pin is what makes reaching for it safe; a test calls a
real tool through it, so a laya release that moves it fails there rather than in
the image.

`LAYA_PRELOAD` and `LAYA_MODELS` are as dead here as they are on the HTTP side:
both only ever chose a preload, and this never preloads, so laya's `_ensure_router`
is never reached. The router comes from the shared `build_router` instead.
"""
import logging
import os

from laya_idle_serve import (
    IdleHook,
    IdleUnloader,
    build_router,
    read_idle_timeout,
    read_max_loaded,
    warn_dead_env,
)

DEFAULT_LOG_LEVEL = "info"


def build_mcp_server(router=None, timeout=None, max_loaded=None):
    """The MCP server, the router behind it, and the watcher that frees it.

    Every argument defaults to the environment and `router` can be handed in so a
    test can drive a real tool with a stand-in. Returns the three pieces because the
    caller owns the watcher's lifetime.
    """
    from laya.mcp import server as laya_mcp

    if max_loaded is None:
        max_loaded = read_max_loaded()
    if timeout is None:
        timeout = read_idle_timeout()
    if router is None:
        router = build_router(max_loaded)
    unloader = IdleUnloader(router, timeout)
    router.add_hook(IdleHook(unloader))
    laya_mcp._ROUTER = router  # the global every tool reads; see the module docstring
    return laya_mcp.server, router, unloader


def main():
    """Serve the tools until stopped, then stop the watcher.

    The port is `LAYA_PORT`, the same variable the HTTP server binds, because both
    answer on the same port inside their own container; Compose publishes this one
    on a different host port.
    """
    from laya.serve import _resolve_port

    # The wrapper logs under the package logger, so that is where the level goes;
    # its lines (cold load, unload, dead env) are the ones worth seeing here.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    level = os.environ.get("LAYA_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    logging.getLogger("laya_idle_serve").setLevel(getattr(logging, level, logging.INFO))
    warn_dead_env()

    server, _router, unloader = build_mcp_server()
    # laya builds its server at import with uvicorn's level fixed at INFO; the image
    # documents LAYA_LOG_LEVEL for both services, so point this one at the same value.
    server.settings.log_level = level
    unloader.start()
    try:
        # json_response and stateless_http together make a tool call one plain JSON
        # POST with no session to negotiate, which is all a tool server on loopback
        # needs. A client that wants the full stateful transport gets it by dropping
        # both flags; the tool behaviour does not change.
        server.run(
            transport="streamable-http",
            host=os.environ.get("LAYA_HOST", "0.0.0.0"),
            port=_resolve_port(),
            json_response=True,
            stateless_http=True,
        )
    finally:
        unloader.stop()


if __name__ == "__main__":
    main()
