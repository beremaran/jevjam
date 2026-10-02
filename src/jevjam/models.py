"""The models behind one router: Laya's Router plus every other backend.

A backend is anything with Laya's Router surface: `loaded`, `route`, `predict`,
`unload(name=None)` and `add_hook`. Laya's Router is the default backend and
answers every request no other backend claims. Other backends also have
`resolve(model)`, which returns their checkpoint name for a client's `model`
field, or None when the request is not theirs.

`Models` puts them behind that same Router surface, so laya's HTTP app, laya's
MCP tools and the idle watcher drive every backend without knowing there is more
than one. It also owns the resident limit: before a request runs, it frees the
least recently used checkpoints, across every backend, until the one the request
needs fits.

Nothing heavy is imported at module level; torch and Julia load with the model.
"""
import gc
import logging
import os
import re
import threading
from types import SimpleNamespace

log = logging.getLogger("jevjam")


class Models:
    """Laya's Router plus other backends, behind the Router's own surface."""

    def __init__(self, default, others=(), max_loaded=2):
        self.default = default
        self.others = list(others)
        self.max_loaded = max_loaded
        self._order = []  # resident checkpoints, least recently used first
        self._lock = threading.Lock()

    @property
    def backends(self):
        return [self.default, *self.others]

    def resolve(self, model):
        """The checkpoint a non-default backend serves for `model`, or None."""
        return next((name for b in self.others if (name := b.resolve(model))), None)

    def _backend(self, model):
        return next((b for b in self.others if b.resolve(model)), self.default)

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

    def predict(self, state, questions=None, model=None, **kwargs):
        backend = self._backend(model)
        self._make_room(backend, backend.route(state, questions, model=model)["model"])
        return backend.predict(state, questions, model=model, **kwargs)

    def unload(self, name=None):
        for b in self.backends:
            if name is None or name in b.loaded:
                b.unload(name)

    def _make_room(self, backend, name):
        """Free least recently used checkpoints until `name` fits under the limit."""
        with self._lock:
            resident = self.loaded
            self._order = [n for n in self._order if n in resident and n != name]
            self._order += [n for n in resident if n not in self._order and n != name]
            while len(self._order) >= self.max_loaded:
                victim = self._order.pop(0)
                log.info("resident limit %d: unloading %s", self.max_loaded, victim)
                self.unload(victim)
            self._order.append(name)


# Keep REVISION equal to the supersonic-julia rev in pyproject.toml, so the code
# and the weights it loads come from the same commit.
REPO = "SupersonicLabs/Julia-1"
REVISION = "a85b127321d580d65176c89ced8273f305745d85"
LINE_ERROR = re.compile(r"^JSONL line (\d+): ")


def _download():
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, allow_patterns=[
        "config.json", "julia_config.json", "model.safetensors", "encoder/*", "tokenizer/*"])


def _device():
    """JEVJAM_DEVICE, or CUDA when there is one; CPU with a warning when CUDA is missing."""
    import torch

    wanted = os.environ.get("JEVJAM_DEVICE", "").strip() or "auto"
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        log.warning("JEVJAM_DEVICE=%s but CUDA is unavailable: Julia runs on CPU", wanted)
        return "cpu"
    return wanted


def load_julia():
    """Download Julia-1 if needed and build it on the configured device."""
    import torch
    from julia import load_model

    # Julia sets torch's global thread count from this, 4 when unset. Without a
    # value it would also cap every Laya checkpoint's CPU threads.
    os.environ.setdefault("JULIA_CPU_THREADS", str(torch.get_num_threads()))
    return load_model(_download(), device=_device(), max_length=8192, head_length=512,
                      strict_encoding=True)


def _rows(state, questions):
    """The rows Julia scores for `questions`; used only to count tokens."""
    def options(q):
        c = q.get("criteria")
        if q.get("type") == "noul":
            return [c["false"], c["true"]] if isinstance(c, dict) else ["false", "true"]
        return list(c.values()) if isinstance(c, dict) else list(c)

    return [dict(state=state, question=q.get("instructions"), type=q.get("type"), options=options(q))
            for q in questions.values()]


def _jev_answer(answer):
    """Julia's answer in Laya's Jev shape: confidence is the chosen option's probability."""
    answer = dict(answer)
    answer.pop("max_probability", None)
    p = answer["probabilities"]
    confidence = max(answer["noul"], 1 - answer["noul"]) if answer["type"] == "noul" else max(p.values())
    return {**answer, "confidence": confidence, "answer_confidence": confidence}


class Julia:
    """SupersonicLabs/Julia-1 as a backend. It answers only when a request names it."""

    name = "julia-1"
    aliases = {"julia-1", "julia", REPO.lower()}

    def __init__(self, loader=load_julia):
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
        return {"model": self.name, "repo": REPO, "reason": "explicit model"}

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

    def predict(self, state, questions=None, model=None, **kwargs):
        engine = self.load()
        # Laya takes a list for choice criteria too; Julia wants label -> description.
        questions = {k: {**q, "criteria": dict(zip(q["criteria"], q["criteria"]))}
                     if q.get("type") == "choice" and isinstance(q.get("criteria"), list) else q
                     for k, q in (questions or {}).items()}
        self._dispatch("on_predict_start")
        try:
            result = engine.predict(state=state, questions=questions)
        except ValueError as error:
            # Julia numbers the question that failed; name it, the way Laya does.
            names = list(questions)
            raise ValueError(LINE_ERROR.sub(
                lambda m: "question %r: " % names[int(m.group(1)) - 1], str(error))) from None
        finally:
            self._dispatch("on_predict_end")
        tokens = sum(row["tokens"] for row in engine.encoding_info(_rows(state, questions)))
        return {
            "model": self.name,
            "answers": {k: _jev_answer(a) for k, a in result["answers"].items()},
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "routing": self.route(state, questions),
        }
