"""Tests for the clef-flash backend, media input, body limits and the shared queue.

No GPU and no download: the engine is a stand-in that records the request it was
given, and the load chain runs against fake weights and a fake free-VRAM figure.
"""
import base64
import io
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from laya.mcp import server as laya_mcp
from PIL import Image
from test_julia import FakeEngine as FakeJuliaEngine
from test_server import MANAGED_ENV, StubRouter

from jevjam import build_app, clef
from jevjam.clef import Clef
from jevjam.models import Julia, ModelUnavailable

STATE = "Is the kettle in this photo switched on?"
QUESTIONS = {
    "on": {"type": "noul", "instructions": "Is the kettle on?"},
    "room": {"type": "choice", "instructions": "Which room?", "criteria": {"kitchen": "a kitchen", "office": "an office"}},
}
GB = 1024 ** 3


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class FakeEngine:
    """clef-flash's engine surface: one Jev request in, one answer out."""

    device = "cpu"

    def __init__(self):
        self.requests = []
        self.videos = []

    def __call__(self, request):
        self.requests.append(request)
        # The video files only exist during the call, so read them now.
        self.videos = [Path(v).read_bytes() if not v.startswith("http") else v for v in request.get("videos", [])]
        return {
            "model": request["model"],
            "answers": {
                "on": {"type": "noul", "noul": 0.9},
                "room": {"type": "choice", "choice": "kitchen", "confidence": 0.8,
                         "probabilities": {"kitchen": 0.8, "office": 0.2}},
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(laya_mcp, "_ROUTER", None, raising=False)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def server(engine):
    router = StubRouter()
    app, models, idle = build_app(router=router, timeout=60,
                                  others=[Julia(loader=FakeJuliaEngine), Clef(loader=lambda: engine)])
    with TestClient(app) as client:
        yield client, router, models


def ask(client, model="clef-flash", **extra):
    return client.post("/v1/systemone", json={"model": model, "state": STATE, "questions": QUESTIONS, **extra})


# ------------------------------------------------------------------------- answers
def test_a_request_that_names_clef_gets_a_jev_shaped_answer(server):
    client, router, models = server
    response = ask(client, "Cloudflare/clef-flash")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "clef-flash"
    assert body["routing"]["repo"] == "Cloudflare/clef-flash"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["answers"]["on"]["answer_confidence"] == pytest.approx(0.9)
    assert body["answers"]["room"]["confidence"] == body["answers"]["room"]["answer_confidence"] == 0.8
    assert router.loads == [] and models.loaded == ["clef-flash"]


def test_clef_swaps_with_the_other_models(server):
    client, _router, models = server
    for model, resident in [(None, "english"), ("clef", "clef-flash"), ("julia-1", "julia-1"), ("clef", "clef-flash")]:
        assert ask(client, model).status_code == 200
        assert models.loaded == [resident]


def test_list_criteria_become_label_descriptions(server, engine):
    client, _router, _models = server
    questions = {"room": {"type": "choice", "instructions": "Which room?", "criteria": ["kitchen", "office"]}}
    assert client.post("/v1/systemone", json={"model": "clef", "state": STATE, "questions": questions}).status_code == 200
    assert engine.requests[0]["questions"]["room"]["criteria"] == {"kitchen": "kitchen", "office": "office"}


# --------------------------------------------------------------------------- media
@pytest.mark.parametrize("encode", [lambda b: b, lambda b: "data:image/png;base64," + b])
def test_base64_images_reach_clef_as_images(server, engine, encode):
    client, _router, _models = server
    assert ask(client, images=[encode(png())]).status_code == 200
    (image,) = engine.requests[0]["images"]
    assert image.size == (4, 4)


def test_image_and_video_urls_pass_through_for_the_processor_to_fetch(server, engine):
    client, _router, _models = server
    assert ask(client, images=["https://example.com/a.png"], videos=["http://example.com/v.mp4"]).status_code == 200
    assert engine.requests[0]["images"] == ["https://example.com/a.png"]
    assert engine.requests[0]["videos"] == ["http://example.com/v.mp4"]


def test_a_base64_video_reaches_clef_as_a_temporary_file(server, engine):
    client, _router, _models = server
    assert ask(client, videos=[base64.b64encode(b"not really an mp4").decode()]).status_code == 200
    assert engine.videos == [b"not really an mp4"]
    assert not Path(engine.requests[0]["videos"][0]).exists(), "the file is gone after the request"


@pytest.mark.parametrize("field, value, message", [
    ("images", ["/etc/passwd"], "images[0] must be base64"),
    ("images", ["file:///etc/passwd"], "images[0] must be base64"),
    ("videos", ["/etc/passwd"], "videos[0] must be base64"),
    ("images", [base64.b64encode(b"hello").decode()], "images[0] is not an image"),
    ("images", [7], "images[0] must be a string"),
    ("images", "https://example.com/a.png", "images must be a list"),
])
def test_paths_and_garbage_are_refused(server, engine, field, value, message):
    client, _router, _models = server
    response = ask(client, **{field: value})
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert engine.requests == [], "nothing reaches the processor"


@pytest.mark.parametrize("model", [None, "julia-1"])
def test_media_for_a_model_that_cannot_see_it_is_refused(server, model):
    client, router, models = server
    response = ask(client, model, images=[png()])
    assert response.status_code == 422
    assert "need clef-flash" in response.json()["detail"]
    assert models.loaded == []


# ---------------------------------------------------------------------- body sizes
def test_only_clef_accepts_bodies_past_two_mib(server):
    client, _router, _models = server
    big = "A" * (3 * 1024 * 1024)  # valid base64, not an image
    assert ask(client, None, padding=big).status_code == 413
    assert ask(client, "julia-1", padding=big).status_code == 413
    assert ask(client, "clef", padding=big).status_code == 200
    assert ask(client, "clef", padding="A" * (21 * 1024 * 1024)).status_code == 413


# ----------------------------------------------------------------------- the queue
def test_requests_from_every_door_run_one_at_a_time():
    running, peak, lock = [0], [0], threading.Lock()

    class SlowEngine(FakeEngine):
        def __call__(self, request):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            return super().__call__(request)

    _app, models, _idle = build_app(router=StubRouter(), timeout=60, others=[Clef(loader=SlowEngine)])
    threads = [threading.Thread(target=models.predict, args=(STATE, QUESTIONS), kwargs={"model": "clef"})
               for _ in range(4)]
    threads.append(threading.Thread(target=laya_mcp.laya_predict_tool,
                                    kwargs={"state": {"q": STATE}, "questions": QUESTIONS, "model": "clef-flash"}))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_mcp_predict_passes_images_to_clef(engine):
    build_app(router=StubRouter(), timeout=60, others=[Clef(loader=lambda: engine)])
    tool = laya_mcp.server._tool_manager.get_tool("jevjam_predict")
    answer = tool.fn(state={"q": STATE}, questions=QUESTIONS, model="clef-flash", images=[png()])
    assert '"clef-flash"' in answer
    assert engine.requests[0]["images"][0].size == (4, 4)


# -------------------------------------------------------------------- the load chain
@pytest.fixture
def chain(monkeypatch, tmp_path):
    """load_clef against 19 GB of fake weights; returns the modes it tried."""
    for i in range(4):
        with open(tmp_path / ("model-%d-of-00004.safetensors" % i), "wb") as f:
            f.truncate(int(4.75 * GB))  # sparse: costs no disk
    tried = []
    state = {"free": 16 * GB, "oom": set()}

    def build(path, mode, dev):
        tried.append(mode)
        if mode in state["oom"]:
            raise torch.cuda.OutOfMemoryError("fake")
        return FakeModel(), None

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(clef, "device", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (state["free"], 16 * GB))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(clef, "_build", build)
    return tried, state


class FakeModel:
    head = torch.nn.Linear(1, 1)


def test_bf16_when_it_fits(chain):
    tried, state = chain
    state["free"] = 40 * GB
    assert clef.load_clef().mode == "bf16"


def test_8bit_on_a_16_gb_card(chain):
    tried, _state = chain
    assert clef.load_clef().mode == "8bit"
    assert tried == ["8bit"], "bf16 is skipped without trying it"


def test_running_out_of_memory_falls_through(chain):
    tried, state = chain
    state["oom"] = {"8bit", "4bit"}
    assert clef.load_clef().mode == "offload"
    assert tried == ["8bit", "4bit", "offload"]


def test_a_forced_mode_is_tried_even_if_it_looks_too_big(chain, monkeypatch):
    tried, _state = chain
    monkeypatch.setenv("JEVJAM_CLEF_QUANT", "bf16")
    assert clef.load_clef().mode == "bf16"


def test_nothing_fits_is_a_clear_error(chain):
    _tried, state = chain
    state["oom"] = set(clef.MODES)
    with pytest.raises(ModelUnavailable, match="clef-flash does not fit: 16.0 GB free on cuda; "
                                               "bf16 needs ~21 GB; 8bit ran out of memory"):
        clef.load_clef()


def test_a_bad_quant_setting_is_a_clear_error(chain, monkeypatch):
    monkeypatch.setenv("JEVJAM_CLEF_QUANT", "3bit")
    with pytest.raises(ModelUnavailable, match="JEVJAM_CLEF_QUANT must be"):
        clef.load_clef()


def test_a_model_that_does_not_fit_answers_503():
    def unavailable():
        raise ModelUnavailable("clef-flash does not fit: 4.0 GB free on cuda")

    app, _models, _idle = build_app(router=StubRouter(), timeout=60, others=[Clef(loader=unavailable)])
    with TestClient(app) as client:
        response = ask(client)
    assert response.status_code == 503
    assert response.json()["detail"] == "clef-flash does not fit: 4.0 GB free on cuda"


def test_vendored_code_and_weights_come_from_one_commit():
    vendored = Path(clef.__file__).parent / "vendor" / "joint_schema_model.py"
    assert clef.REVISION in vendored.read_text().splitlines()[1]
