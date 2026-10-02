"""Jev-compatible HTTP and MCP servers for Laya and Julia, freed when idle.

Laya's own server answers with Laya alone. This one also answers with
SupersonicLabs/Julia-1 when a request names it (see `models.py`), and puts both
behind one resident limit and one idle timer.

`laya-serve` loads the checkpoints at boot and then holds them for the life of the
process. That is the wrong shape for a server that sits unused between bursts: the
GPU memory stays committed to checkpoints nobody is asking anything.

This wrapper is the same server with one difference. It never preloads, so a fresh
container holds no checkpoint at all, and it frees every checkpoint once no request
has reached the router for `JEVJAM_IDLE_TIMEOUT` seconds. Waking costs nothing extra:
`Router.predict` already loads the checkpoint a request routes to, so the first
request after a sleep builds one checkpoint, not three.

    JEVJAM_IDLE_TIMEOUT   seconds of no inference before unloading; 0 disables the
                          timer and the checkpoints stay resident (default 300)
    JEVJAM_MAX_LOADED     checkpoints, across Laya and Julia, that may stay resident
                          while the server is awake (default 2)

Both are whole numbers, and a value that is not one is a startup error naming the
variable, rather than a timeout nobody notices.

Every `JEVJAM_*` setting used to be `LAYA_*`. The old name still works with a
warning; `apply_env` copies it across, then copies the new names back to the
`LAYA_*` ones laya reads itself.

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

from .models import Julia, Models

log = logging.getLogger("jevjam")

DEFAULT_IDLE_TIMEOUT = 300  # seconds
DEFAULT_MAX_LOADED = 2      # laya's own default
DEFAULT_PORT = 8000

# Only checked while nothing is resident, so an armed watcher costs nothing and a
# sleeping one wakes about once a second to notice a request.
POLL_SECONDS = 1.0

# laya.serve reads these to choose a preload. There is no preload, so honouring them
# would be a lie; they are only worth a warning.
DEAD_ENV = ("LAYA_PRELOAD", "LAYA_MODELS")

# Settings that moved from LAYA_X to JEVJAM_X, and the ones laya still reads as LAYA_X.
RENAMED_ENV = ("HOST", "PORT", "DEVICE", "API_KEY", "LOG_LEVEL", "IDLE_TIMEOUT", "MAX_LOADED", "THREADS")
LAYA_READS = ("HOST", "PORT", "DEVICE", "API_KEY", "LOG_LEVEL", "THREADS")


def _env(name):
    return os.environ.get(name, "").strip()


def apply_env():
    """Accept the old LAYA_* names, then hand the JEVJAM_* values to laya and Julia.

    A blank value counts as unset, since Compose passes an unset passthrough as an
    empty string. Safe to call twice: the second call finds JEVJAM_* already set.
    """
    for key in RENAMED_ENV:
        new, old = "JEVJAM_" + key, "LAYA_" + key
        if not _env(new) and _env(old):
            log.warning("%s is deprecated; use %s", old, new)
            os.environ[new] = os.environ[old]
        if key in LAYA_READS and _env(new):
            os.environ[old] = os.environ[new]
    if _env("JEVJAM_THREADS"):
        os.environ["JULIA_CPU_THREADS"] = os.environ["JEVJAM_THREADS"]


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
    return _int_env("JEVJAM_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT, 0)


def read_max_loaded():
    """How many checkpoints may stay resident while the server is awake."""
    return _int_env("JEVJAM_MAX_LOADED", DEFAULT_MAX_LOADED, 1)


def warn_dead_env():
    """Say once, at startup, that a variable which used to work no longer does."""
    set_but_ignored = [name for name in DEAD_ENV if os.environ.get(name, "").strip()]
    if set_but_ignored:
        log.warning(
            "ignoring %s: this server starts cold and keeps its checkpoints until "
            "JEVJAM_IDLE_TIMEOUT expires", " and ".join(set_but_ignored))


def build_router(max_loaded):
    """The Router laya.serve would build, minus the preload and plus `max_loaded`.

    `laya.serve.build_router` cannot be used: it preloads unless LAYA_PRELOAD says
    otherwise, and it has no way to pass `max_loaded`. Its helpers are imported
    rather than copied so device, thread and LAYA_AUTO_TASK settings keep behaving
    exactly as laya documents them; `apply_env` has already copied the JEVJAM_*
    values into the LAYA_* names they read. The `laya[serve]==0.3.20` pin is what
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
        self._thread = threading.Thread(target=self._run, name="jevjam-idle", daemon=True)
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


def rename_mcp(laya_mcp):
    """Serve laya's MCP tools as jevjam_* from a server named jevjam."""
    manager = laya_mcp.server._tool_manager
    laya_mcp.server._lowlevel_server.name = "jevjam"
    for old in ("laya_predict", "laya_preset", "laya_route", "laya_status"):
        tool = manager.get_tool(old)
        if tool is None:  # already renamed by an earlier build_app
            continue
        description = tool.description
        if old == "laya_predict":
            description += (" model: 'auto' (default), 'english', 'multilingual', "
                            "'typed-decisions', or 'julia-1' for SupersonicLabs/Julia-1.")
        laya_mcp.server.remove_tool(old)
        laya_mcp.server.add_tool(tool.fn, name="jevjam_" + old[len("laya_"):], title=tool.title,
                                 description=description, annotations=tool.annotations)


def build_app(router=None, timeout=None, max_loaded=None, others=None):
    """The HTTP and MCP ASGI app, its shared models, and its idle watcher.

    Every argument defaults to the environment. `router` is laya's Router and
    `others` the other backends; tests hand in stand-ins for both. The returned
    `Models` is the one object both endpoints and the watcher use. The app's
    lifespan owns the MCP session manager and watcher; the returned watcher is
    exposed for tests and status.
    """
    from contextlib import asynccontextmanager
    from fastapi.responses import JSONResponse
    from laya import serve as laya_serve
    from laya.mcp import server as laya_mcp
    from laya.mcp import tools as laya_tools
    from mcp.server.transport_security import TransportSecuritySettings

    apply_env()
    if max_loaded is None:
        max_loaded = read_max_loaded()
    if timeout is None:
        timeout = read_idle_timeout()
    if router is None:
        router = build_router(max_loaded)
    if others is None:
        others = [Julia()]
    models = Models(router, others, max_loaded)
    router = models
    mcp_api_key = _env("JEVJAM_API_KEY") or None

    # laya drops a `model` it does not know before the request reaches the router.
    # Let the other backends claim theirs first, on both endpoints.
    laya_resolve = getattr(laya_serve._resolve_model, "laya", laya_serve._resolve_model)

    def resolve_model(model):
        return models.resolve(model) or laya_resolve(model)

    resolve_model.laya = laya_resolve
    laya_serve._resolve_model = resolve_model
    laya_tools.VALID_MODELS.update(alias for b in others for alias in b.aliases)
    rename_mcp(laya_mcp)
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
    laya_mcp.server.settings.log_level = (_env("JEVJAM_LOG_LEVEL") or "info").upper()

    mcp_app = laya_mcp.server.streamable_http_app(
        json_response=True,
        stateless_http=True,
        # MCP SDK defaults to localhost-only Host checks. The top-level server is
        # loopback-bound by Compose; keep the endpoint usable through proxy hosts too.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    session_manager = laya_mcp.server.session_manager

    app = laya_serve.create_app(router)
    app.state.mcp_ready = False

    # Keep laya's health fields while adding readiness for the mounted MCP service.
    @app.middleware("http")
    async def add_mcp_readiness(request, call_next):
        path = request.scope["path"]
        if request.method == "GET" and path == "/health":
            return JSONResponse({
                "status": "ok",
                "loaded": router.loaded,
                "device": _env("JEVJAM_DEVICE") or "auto",
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

    # basicConfig leaves the root logger alone if something already configured one,
    # and uvicorn sets up its own loggers, so this only decides our own lines.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    apply_env()
    level = _env("JEVJAM_LOG_LEVEL") or "info"
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    warn_dead_env()

    app, _router, _unloader = build_app()
    uvicorn.run(
        app,
        host=_env("JEVJAM_HOST") or "0.0.0.0",
        port=_int_env("JEVJAM_PORT", DEFAULT_PORT, 1),
        log_level=level,
    )


if __name__ == "__main__":
    main()
