"""The models behind one router: Laya's Router plus every other backend.

A backend is anything with Laya's Router surface: `loaded`, `route`, `predict`,
`unload(name=None)` and `add_hook`. Laya's Router is the default backend and
answers every request no other backend claims. Other backends subclass `Backend`
and add `resolve(model)`, which returns their checkpoint name for a client's
`model` field, or None when the request is not theirs.

`Models` puts them behind that same Router surface, so the HTTP app, laya's MCP
tools and the idle watcher drive every backend without knowing there is more than
one. It runs every inference and unload through one FIFO queue, and keeps one
checkpoint resident: before a request runs, it frees every other one.

Nothing heavy is imported at module level; torch and each model load on first use.
"""
import gc
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

log = logging.getLogger("jevjam")

TEXT_BODY_BYTES = 2 * 1024 * 1024  # laya.serve's own cap


class ModelUnavailable(RuntimeError):
    """The model cannot run on this machine. The message is safe to show a client."""


class Models:
    """Laya's Router plus other backends, behind the Router's own surface."""

    def __init__(self, default, others=()):
        self.default = default
        self.others = list(others)
        # One worker: requests from HTTP and MCP run in arrival order, one at a time,
        # so a load never races another request's inference.
        self._queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jevjam-queue")

    @property
    def backends(self):
        return [self.default, *self.others]

    def resolve(self, model):
        """The checkpoint a non-default backend serves for `model`, or None."""
        return next((name for b in self.others if (name := b.resolve(model))), None)

    def _backend(self, model):
        return next((b for b in self.others if b.resolve(model)), self.default)

    def max_body_bytes(self, model):
        """The largest request body the backend serving `model` accepts."""
        return getattr(self._backend(model), "max_body_bytes", TEXT_BODY_BYTES)

    @property
    def largest_body_bytes(self):
        return max(getattr(b, "max_body_bytes", TEXT_BODY_BYTES) for b in self.backends)

    @property
    def loaded(self):
        return [name for b in self.backends for name in b.loaded]

    @property
    def _agents(self):
        # laya.mcp reads this to report the device each checkpoint runs on.
        agents = {}
        for b in self.backends:
            agents.update(getattr(b, "_agents", None) or {})
        return agents

    def add_hook(self, hook):
        for b in self.backends:
            b.add_hook(hook)
        return self

    def route(self, state, questions=None, model=None, **kwargs):
        return self._backend(model).route(state, questions, model=model, **kwargs)

    def predict(self, state, questions=None, model=None, images=None, videos=None, **kwargs):
        return self._queue.submit(self._predict, state, questions, model, images, videos, kwargs).result()

    def unload(self, name=None):
        return self._queue.submit(self._unload, name).result()

    def _predict(self, state, questions, model, images, videos, kwargs):
        backend = self._backend(model)
        if images or videos:
            if not getattr(backend, "media", False):
                raise ValueError("images and videos need clef-flash; set model to clef-flash")
            kwargs = {**kwargs, "images": images, "videos": videos}
        name = backend.route(state, questions, model=model)["model"]
        for other in self.loaded:
            if other != name:
                log.info("unloading %s to make room for %s", other, name)
                self._unload(other)
        return backend.predict(state, questions, model=model, **kwargs)

    def _unload(self, name=None):
        for b in self.backends:
            if name is None or name in b.loaded:
                b.unload(name)


class Backend:
    """A model that answers only requests naming it. Subclasses set `name`,
    `aliases` and `repo`, and implement `_answer(engine, state, questions, **media)`."""

    def __init__(self, loader):
        self._loader = loader
        self._engine = None
        self._hooks = []
        self._lock = threading.Lock()

    def resolve(self, model):
        return self.name if str(model or "").strip().lower() in self.aliases else None

    @property
    def loaded(self):
        return [self.name] if self._engine is not None else []

    @property
    def _agents(self):
        return {self.name: self._engine} if self._engine is not None else {}

    def add_hook(self, hook):
        self._hooks.append(hook)
        return self

    def _dispatch(self, event):
        ctx = SimpleNamespace(router=self, model=self.name)
        for hook in self._hooks:
            method = getattr(hook, event, None)
            if method is not None:
                method(ctx)

    def route(self, state, questions=None, model=None, **kwargs):
        return {"model": self.name, "repo": self.repo, "reason": "explicit model"}

    def load(self):
        with self._lock:
            if self._engine is None:
                self._engine = self._loader()
                self._dispatch("on_load")
            return self._engine

    def unload(self, name=None):
        with self._lock:
            self._engine = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def predict(self, state, questions=None, model=None, **media):
        # Laya takes a list for choice criteria too; the others want label -> description.
        questions = {k: {**q, "criteria": dict(zip(q["criteria"], q["criteria"]))}
                     if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), list)
                     else q for k, q in (questions or {}).items()}
        self._dispatch("on_predict_start")
        try:
            answers, usage = self._answer(self.load(), state, questions, **media)
        finally:
            self._dispatch("on_predict_end")
        return {
            "model": self.name,
            "answers": {k: jev_answer(a) for k, a in answers.items()},
            "usage": usage,
            "routing": self.route(state, questions),
        }


def jev_answer(answer):
    """An answer in Laya's Jev shape: confidence is the chosen option's probability."""
    answer = dict(answer)
    answer.pop("max_probability", None)
    if answer["type"] == "noul":
        confidence = max(answer["noul"], 1 - answer["noul"])
    else:
        confidence = answer.get("confidence", max(answer["probabilities"].values()))
    return {**answer, "confidence": confidence, "answer_confidence": confidence}


def device():
    """JEVJAM_DEVICE, or CUDA when there is one; CPU with a warning when CUDA is missing."""
    import torch

    wanted = os.environ.get("JEVJAM_DEVICE", "").strip() or "auto"
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        log.warning("JEVJAM_DEVICE=%s but CUDA is unavailable: running on CPU", wanted)
        return "cpu"
    return wanted


# Keep REVISION equal to the supersonic-julia rev in pyproject.toml, so the code
# and the weights it loads come from the same commit.
JULIA_REPO = "SupersonicLabs/Julia-1"
JULIA_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
LINE_ERROR = re.compile(r"^JSONL line (\d+): ")


def load_julia():
    """Download Julia-1 if needed and build it on the configured device."""
    import julia.router.encoder
    import torch
    from huggingface_hub import snapshot_download
    from julia import load_model

    # Julia's fast ModernBERT path calls a private method that transformers 5.1
    # removed. Stock ModernBERT gives the same probabilities, about 1 ms slower.
    julia.router.encoder.specialize_decision_encoder = lambda model: False
    # Julia sets torch's global thread count from this, 4 when unset. Without a
    # value it would also cap every Laya checkpoint's CPU threads.
    os.environ.setdefault("JULIA_CPU_THREADS", str(torch.get_num_threads()))
    path = snapshot_download(JULIA_REPO, revision=JULIA_REVISION, allow_patterns=[
        "config.json", "julia_config.json", "model.safetensors", "encoder/*", "tokenizer/*"])
    return load_model(path, device=device(), max_length=8192, head_length=512, strict_encoding=True)


def _julia_rows(state, questions):
    """The rows Julia scores for `questions`; used only to count tokens."""
    def options(q):
        c = q.get("criteria")
        if q.get("type") == "noul":
            return [c["false"], c["true"]] if isinstance(c, dict) else ["false", "true"]
        return list(c.values()) if isinstance(c, dict) else list(c)

    return [dict(state=state, question=q.get("instructions"), type=q.get("type"), options=options(q))
            for q in questions.values()]


class Julia(Backend):
    """SupersonicLabs/Julia-1, a 144M ModernBERT decision model."""

    name = "julia-1"
    aliases = {"julia-1", "julia", JULIA_REPO.lower()}
    repo = JULIA_REPO

    def __init__(self, loader=load_julia):
        super().__init__(loader)

    def _answer(self, engine, state, questions):
        try:
            result = engine.predict(state=state, questions=questions)
        except ValueError as error:
            # Julia numbers the question that failed; name it, the way Laya does.
            names = list(questions)
            raise ValueError(LINE_ERROR.sub(
                lambda m: "question %r: " % names[int(m.group(1)) - 1], str(error))) from None
        tokens = sum(row["tokens"] for row in engine.encoding_info(_julia_rows(state, questions)))
        return result["answers"], {"input_tokens": tokens, "output_tokens": 0}
