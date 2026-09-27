"""Laya's HTTP and MCP servers, with checkpoints freed when they go idle.

`laya-serve` loads the checkpoints at boot and then holds them for the life of the
process. That is the wrong shape for a server that sits unused between bursts: the
GPU memory stays committed to checkpoints nobody is asking anything.

This wrapper is the same server with one difference. It never preloads, so a fresh
container holds no checkpoint at all, and it frees every checkpoint once no request
has reached the router for `LAYA_IDLE_TIMEOUT` seconds. Waking costs nothing extra:
`Router.predict` already loads the checkpoint a request routes to, so the first
request after a sleep builds one checkpoint, not three.

    LAYA_IDLE_TIMEOUT   seconds of no inference before unloading; 0 disables the
                        timer and the checkpoints stay resident (default 300)
    LAYA_MAX_LOADED     checkpoints that may stay resident while the server is awake,
                        passed to `Router(max_loaded=...)` (default 2)

Both are whole numbers of seconds, and a value that is not one is a startup error
naming the variable, rather than a timeout nobody notices.

`LAYA_PRELOAD` and `LAYA_MODELS` no longer do anything: both only ever fed a preload,
and there is no preload. Setting either logs a warning and is otherwise ignored.

Nothing heavy is imported at module level. `import laya` and `import laya.serve` are
both cheap and torch-free, and the config and timer code below stays testable
without a GPU, following the same rule laya sets for itself.
"""
import hmac
import logging
import os
import threading
import time

log = logging.getLogger("laya_idle_serve")

DEFAULT_IDLE_TIMEOUT = 300  # seconds
DEFAULT_MAX_LOADED = 2      # laya's own default; the two checkpoints routing can pick

# Only checked while nothing is resident, so an armed watcher costs nothing and a
# sleeping one wakes about once a second to notice a request.
POLL_SECONDS = 1.0

# laya.serve reads these to choose a preload. There is no preload, so honouring them
# would be a lie; they are only worth a warning.
DEAD_ENV = ("LAYA_PRELOAD", "LAYA_MODELS")


def _int_env(name, default, minimum):
    """A whole number from the environment, or `default` when unset or blank.

    A blank value is not an error: Compose passes an unset passthrough as an empty
    string, and that has to mean "use the default".
    """
    raw = os.environ.get(name, "")
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
        if value < minimum:
            raise ValueError(text)
    except ValueError:
        raise SystemExit("invalid %s %r: must be a whole number of %d or more" % (name, raw, minimum))
    return value


def read_idle_timeout():
    """Seconds of silence before the checkpoints are freed. 0 never frees them."""
    return _int_env("LAYA_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT, 0)


def read_max_loaded():
    """How many checkpoints may stay resident while the server is awake."""
    return _int_env("LAYA_MAX_LOADED", DEFAULT_MAX_LOADED, 1)


def warn_dead_env():
    """Say once, at startup, that a variable which used to work no longer does."""
    set_but_ignored = [name for name in DEAD_ENV if os.environ.get(name, "").strip()]
    if set_but_ignored:
        log.warning(
            "ignoring %s: this server starts cold and keeps its checkpoints until "
            "LAYA_IDLE_TIMEOUT expires", " and ".join(set_but_ignored))


def build_router(max_loaded):
    """The Router laya.serve would build, minus the preload and plus `max_loaded`.

    `laya.serve.build_router` cannot be used: it preloads unless LAYA_PRELOAD says
    otherwise, and it has no way to pass `max_loaded`. Its helpers are imported
    rather than copied so LAYA_DEVICE, LAYA_AUTO_TASK and LAYA_THREADS keep
    behaving exactly as laya documents them. The `laya[serve]==0.3.20` pin is what
    makes reaching for them safe; a test builds a real Router through this function,
    so a laya release that moves them fails there rather than in the image.
    """
    from laya.router import Router
    from laya.serve import _apply_thread_limit, _env_bool

    _apply_thread_limit()
    return Router(
        device=os.environ.get("LAYA_DEVICE") or None,
        auto_task_detection=_env_bool("LAYA_AUTO_TASK", False),
        max_loaded=max_loaded,
    )


def _valid_bearer(authorization, key):
    scheme, separator, token = authorization.partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))
    )


class IdleUnloader:
    """Frees every checkpoint once no request has reached the router for `timeout`.

    One daemon thread owns the policy. It sleeps until the deadline rather than
    polling, stamps come from the router hooks, and it disarms after unloading so it
    waits for the next request instead of spinning on a router with nothing loaded.

    A request stamps the clock when it starts as well as when it ends, so a request
    in flight is never the reason the thread unloads: the model it is using stays
    reachable through the reference `Router.predict` already holds.
    """

    def __init__(self, router, timeout):
        self._router = router
        self._timeout = float(timeout)
        self._last = None  # None until the first request, and again after unloading
        self._stop = threading.Event()
        self._thread = None

    def touch(self):
        """Record activity. Called at the start and the end of every inference."""
        self._last = time.monotonic()

    @property
    def last_activity(self):
        """When a request last reached the router, or None while nothing is loaded."""
        return self._last

    def start(self):
        """Run the watcher. A timeout of 0 means the checkpoints stay resident."""
        if not self._timeout:
            return
        self._thread = threading.Thread(target=self._run, name="laya-idle", daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the watcher and wait for it, so the process does not outlive it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            last = self._last
            if last is None:
                self._stop.wait(POLL_SECONDS)  # asleep: nothing to free, nothing to time
                continue
            wait = self._timeout - (time.monotonic() - last)
            if wait > 0:
                self._stop.wait(wait)          # a new request resets the clock and we recompute
                continue
            freed = self._router.loaded
            if freed:
                self._router.unload()
                log.info("no request for %.0fs: unloaded %s", self._timeout, ", ".join(freed))
            self._last = None


class IdleHook:
    """Stamps the idle clock and logs the two moments worth seeing in the log.

    Stamping on both ends of a prediction is what keeps the watcher from unloading
    under a request that is still running: the start stamp moves the deadline out for
    the whole of the forward pass, not just up to the moment the request arrived.

    The only line logged here is `on_load`'s; the unload line comes from the watcher
    that does the unloading. `on_route` runs for route-only calls too -- `laya_route`
    on the MCP endpoint promises no forward pass -- so a load line hung off it would
    announce a load that never happens.
    """

    def __init__(self, unloader):
        self._unloader = unloader

    def on_predict_start(self, ctx):
        self._unloader.touch()

    def on_predict_end(self, ctx):
        self._unloader.touch()

    def on_load(self, ctx):
        log.info("loaded %s", ctx.model)


def build_app(router=None, timeout=None, max_loaded=None):
    """The HTTP and MCP ASGI app, its shared router, and its idle watcher.

    Every argument defaults to the environment, and `router` can be handed in so a
    test can drive both endpoints with a stand-in. The app's lifespan owns the MCP
    session manager and watcher; the returned watcher is exposed for tests and status.
    """
    from contextlib import asynccontextmanager
    from fastapi.responses import JSONResponse
    from laya.serve import create_app
    from laya.mcp import server as laya_mcp
    from mcp.server.transport_security import TransportSecuritySettings

    if max_loaded is None:
        max_loaded = read_max_loaded()
    if timeout is None:
        timeout = read_idle_timeout()
    if router is None:
        router = build_router(max_loaded)
    mcp_api_key = os.environ.get("LAYA_API_KEY") or None
    unloader = IdleUnloader(router, timeout)
    router.add_hook(IdleHook(unloader))

    # Laya's MCP tools use this module global. Do not let a missed assignment fall
    # back to laya's lazy builder: that would create a second Router outside the
    # idle watcher and could load another copy of a checkpoint into VRAM.
    laya_mcp._ROUTER = router

    def require_shared_router():
        if laya_mcp._ROUTER is None:
            raise laya_mcp.ToolError("internal_error", "the shared router is not configured")
        return laya_mcp._ROUTER

    laya_mcp._ensure_router = require_shared_router
    laya_mcp.server.settings.log_level = os.environ.get("LAYA_LOG_LEVEL", "info").upper()

    mcp_app = laya_mcp.server.streamable_http_app(
        json_response=True,
        stateless_http=True,
        # MCP SDK defaults to localhost-only Host checks. The top-level server is
        # loopback-bound by Compose; keep the endpoint usable through proxy hosts too.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    session_manager = laya_mcp.server.session_manager

    app = create_app(router)
    app.state.mcp_ready = False

    # Keep laya's health fields while adding readiness for the mounted MCP service.
    @app.middleware("http")
    async def add_mcp_readiness(request, call_next):
        path = request.scope["path"]
        if request.method == "GET" and path == "/health":
            return JSONResponse({
                "status": "ok",
                "loaded": router.loaded,
                "device": os.environ.get("LAYA_DEVICE") or "auto",
                "mcp_ready": app.state.mcp_ready,
            })
        if mcp_api_key and (path == "/mcp" or path.startswith("/mcp/")):
            if not _valid_bearer(request.headers.get("authorization", ""), mcp_api_key):
                return JSONResponse(
                    {"detail": "Unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    app.mount("/", mcp_app)
    http_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(application):
        try:
            async with session_manager.run():
                async with http_lifespan(application):
                    unloader.start()
                    application.state.mcp_ready = True
                    try:
                        yield
                    finally:
                        application.state.mcp_ready = False
                        unloader.stop()
        finally:
            application.state.mcp_ready = False

    app.router.lifespan_context = combined_lifespan
    return app, router, unloader


def main():
    """Run the server until it is stopped, then stop the watcher."""
    import uvicorn
    from laya.serve import _resolve_port

    # basicConfig leaves the root logger alone if something already configured one,
    # and uvicorn sets up its own loggers, so this only decides our own lines.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    log.setLevel(getattr(logging, os.environ.get("LAYA_LOG_LEVEL", "info").upper(), logging.INFO))
    warn_dead_env()

    app, _router, _unloader = build_app()
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
