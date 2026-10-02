"""Cloudflare/clef-flash: a 9B Qwen3.5 decision model that also reads images and video.

Its 19 GB of BF16 weights do not fit every GPU, so loading walks a chain: BF16 if
it fits in free VRAM, else 8-bit, else 4-bit, else split across GPU and CPU. A mode
that runs out of memory anyway falls through to the next one, and when none loads
the request gets a `ModelUnavailable` naming what was tried. `JEVJAM_CLEF_QUANT`
forces one mode.

Media comes as base64, a `data:` URI or an http(s) URL, which the server fetches.
Anything else is refused: given any other string, the Hugging Face processor would
open it as a path on the server.
"""
import base64
import gc
import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

from .models import Backend, ModelUnavailable, device

log = logging.getLogger("jevjam")

# Keep REVISION equal to the commit named at the top of vendor/joint_schema_model.py.
REPO = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
MODES = ("bf16", "8bit", "4bit", "offload")
SHARE = {"bf16": 1.0, "8bit": 0.55, "4bit": 0.33}  # of the BF16 weight size
HEADROOM = 2 * 1024 ** 3                           # activations and CUDA context
MEDIA_BODY_BYTES = 20 * 1024 * 1024
GB = 1024 ** 3


def _build(path, mode, dev):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    from .vendor.joint_schema_model import ClefModel, JointSchemaHead

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": dev}}
    keep = ["visual", "lm_head"]  # small, and the head reads lm_head's weights
    if mode == "8bit":
        # LLM.int8 always computes in fp16, and says so on every layer of every request.
        logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=keep)
    elif mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=keep)
    elif mode == "offload":
        kwargs["device_map"] = _offload_map(path, dev)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, **kwargs)
    backbone.config.use_cache = False
    head = JointSchemaHead(**json.loads((Path(path) / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(Path(path) / "joint_head.safetensors"), strict=True)
    head = head.to(device=dev, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(path)
    _decode_videos_with_pyav(processor.video_processor)
    return ClefModel(backbone, head).eval(), processor


def _decode_videos_with_pyav(video_processor):
    """transformers decodes video with torchcodec, which needs system FFmpeg libraries.
    PyAV ships its own; the processor still picks which frames to sample."""
    from transformers.video_utils import load_video

    def fetch_videos(videos, sample_indices_fn=None):
        if isinstance(videos, list):
            return list(zip(*[fetch_videos(v, sample_indices_fn) for v in videos]))
        return load_video(videos, backend="pyav", sample_indices_fn=sample_indices_fn)

    video_processor.fetch_videos = fetch_videos


def _offload_map(path, dev):
    """As many layers on the GPU as fit, the rest on CPU. lm_head stays on the GPU:
    the head reads its weights directly, which an offloaded module keeps on `meta`."""
    import psutil
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    with init_empty_weights():
        empty = Qwen3_5ForConditionalGeneration._from_config(AutoConfig.from_pretrained(path), dtype=torch.bfloat16)
    gpu = torch.device(dev).index or 0
    lm_head = empty.get_output_embeddings().weight.numel() * 2
    room = max(torch.cuda.mem_get_info(gpu)[0] - HEADROOM - lm_head, 0)
    device_map = infer_auto_device_map(
        empty, max_memory={gpu: room, "cpu": psutil.virtual_memory().available},
        no_split_module_classes=empty._no_split_modules, dtype=torch.bfloat16)
    device_map["lm_head"] = gpu
    return device_map


class Engine:
    """A loaded clef-flash: answers one Jev request body."""

    def __init__(self, model, processor, mode):
        self.model, self.processor, self.mode = model, processor, mode
        self.device = next(model.head.parameters()).device

    def __call__(self, request):
        from .vendor.joint_schema_model import systemone

        return systemone(self.model, self.processor, request)


def load_clef():
    """Download clef-flash if needed and load it in the first mode that fits."""
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, revision=REVISION)
    dev = device()
    if dev == "cpu":
        log.warning("clef-flash on CPU: expect seconds per request")
        return Engine(*_build(path, "bf16", "cpu"), "cpu")

    forced = os.environ.get("JEVJAM_CLEF_QUANT", "").strip().lower()
    if forced and forced != "auto" and forced not in MODES:
        raise ModelUnavailable("JEVJAM_CLEF_QUANT must be auto or one of %s, not %r" % (", ".join(MODES), forced))
    weights = sum(f.stat().st_size for f in Path(path).glob("model-*.safetensors"))
    free = torch.cuda.mem_get_info(torch.device(dev))[0]
    tried = []
    for mode in [forced] if forced in MODES else MODES:
        need = weights * SHARE.get(mode, 0) + HEADROOM
        if forced not in MODES and mode != "offload" and need > free:
            tried.append("%s needs ~%.0f GB" % (mode, need / GB))
            continue
        try:
            engine = Engine(*_build(path, mode, dev), mode)
            log.info("clef-flash loaded as %s", mode)
            return engine
        except torch.cuda.OutOfMemoryError:
            tried.append("%s ran out of memory" % mode)
            gc.collect()
            torch.cuda.empty_cache()
    raise ModelUnavailable("clef-flash does not fit: %.1f GB free on %s; %s"
                           % (free / GB, dev, "; ".join(tried)))


def _decode(item, where):
    data = item.split(",", 1)[-1] if item.startswith("data:") else item
    try:
        return base64.b64decode(data, validate=True)
    except ValueError:
        raise ValueError("%s must be base64, a data: URI or an http(s) URL" % where) from None


def _media(items, kind, tmp):
    """Each item as the processor wants it: URLs as they are, base64 decoded."""
    from PIL import Image, UnidentifiedImageError

    if not isinstance(items, list):
        raise ValueError("%s must be a list" % kind)
    out = []
    for i, item in enumerate(items):
        where = "%s[%d]" % (kind, i)
        if not isinstance(item, str):
            raise ValueError("%s must be a string" % where)
        if item.startswith(("http://", "https://")):
            out.append(item)
        elif kind == "images":
            try:
                out.append(Image.open(BytesIO(_decode(item, where))))
            except UnidentifiedImageError:
                raise ValueError("%s is not an image" % where) from None
        else:
            # The video decoder reads files, so the bytes go to one only we name.
            path = Path(tmp) / ("video-%d" % i)
            path.write_bytes(_decode(item, where))
            out.append(str(path))
    return out


class Clef(Backend):
    """Cloudflare/clef-flash. Also the one backend that takes images and videos."""

    name = "clef-flash"
    aliases = {"clef-flash", "clef", REPO.lower()}
    repo = REPO
    media = True
    max_body_bytes = MEDIA_BODY_BYTES

    def __init__(self, loader=load_clef):
        super().__init__(loader)

    def _answer(self, engine, state, questions, images=None, videos=None):
        request = {"model": self.name, "state": state, "questions": questions}
        with tempfile.TemporaryDirectory(prefix="jevjam-") as tmp:
            if images:
                request["images"] = _media(images, "images", tmp)
            if videos:
                request["videos"] = _media(videos, "videos", tmp)
            result = engine(request)
        return result["answers"], result["usage"]
