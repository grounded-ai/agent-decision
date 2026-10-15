"""agent-decision-3b (or agent-decision-9b, DECIDER_BACKEND=clef) as a small HTTP service at the standard decision-model API,
`POST /v1/systemone`, so any client of that API can use it: grounded-ai's
Evaluator("systemone/agent-decision-3b", base_url="http://127.0.0.1:8765"), CascadeEvaluator, compare, ...

    pip install torch "transformers>=5.10.2" "peft>=0.18" huggingface_hub   # + flash-linear-attention for the 9B
    python serve/decision_server.py                    # agent-decision-3b on 127.0.0.1:8765
    DECIDER_BACKEND=clef python serve/decision_server.py   # agent-decision-9b (CUDA)

    POST /v1/systemone  {"state": ..., "questions": {...}}  ->  {"answers": {...}, "model": ..., "latency_ms": ...}
    POST /decide        the same (kept for older hooks)
    GET  /health        ->  {"ok": true, "model": ..., "device": ...}

Settings: DECIDER_PORT (8765), DECIDER_ADAPTER (a Hub repo or a local folder), DECIDER_REVISION (e.g. v1).

The model is loaded exactly as the model card shows: d1-3B at its pinned revision, the LoRA adapter merged, and the
fitted temperature applied to the option scores at readout. The 9B is Clef-flash at its pinned revision with the
LoRA adapter merged and the calibrated joint head loaded; Clef's own `systemone` answers in the same format.
`confidence` is reported as the base models report it (the top probability); grounded-ai recomputes it with
TypeSafe's definition from the probabilities.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModel

BACKEND = os.environ.get("DECIDER_BACKEND", "d1")  # "d1" (agent-decision-3b) or "clef" (agent-decision-9b)
BASE, BASE_REVISION = {"d1": ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432"),
                       "clef": ("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3")}[BACKEND]
ADAPTER = os.environ.get("DECIDER_ADAPTER", {"d1": "grounded-ai/agent-decision-3b", "clef": "grounded-ai/agent-decision-9b"}[BACKEND])
# ADAPTER: a Hub repo (DECIDER_REVISION: e.g. v1) or a local folder
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Temperature:
    """Divides each option's score by the fitted temperature before the softmax."""

    def __init__(self, t):
        self.t = t

    def apply(self, question, scores):
        return [s / self.t for s in scores]


def adapter_dir():
    return ADAPTER if os.path.isdir(ADAPTER) else snapshot_download(ADAPTER, revision=os.environ.get("DECIDER_REVISION"))


def load():
    """A function (state, questions) -> SystemOne response."""
    if BACKEND == "clef":
        return load_clef()
    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    model = AutoModel.from_pretrained(BASE, revision=BASE_REVISION, trust_remote_code=True, dtype=dtype).to(DEVICE)
    ours = adapter_dir()
    model = PeftModel.from_pretrained(model, ours).merge_and_unload()
    model.engine.calibration = Temperature(json.load(open(f"{ours}/calibration.json"))["temperature"])
    return model.system_one


def load_clef():
    """The 9B as its model card loads it; the temperature is already folded into the head. Needs a CUDA GPU."""
    from safetensors.torch import load_file
    base = snapshot_download(BASE, revision=BASE_REVISION)
    sys.path.insert(0, base)
    import joint_schema_model as jsm
    model, processor = jsm.load_release_model(base, device=DEVICE)
    ours = adapter_dir()
    model.language_model = PeftModel.from_pretrained(model.language_model, ours).merge_and_unload()
    model.head.load_state_dict(load_file(f"{ours}/joint_head.safetensors"))
    model.eval()
    return lambda state, questions: jsm.systemone(model, processor, {"model": "agent-decision-9b", "state": state,
                                                                      "questions": questions})


DECIDE, LOCK = None, Lock()
# The name /v1/systemone responses carry: the adapter's repo name, or the folder's name.
MODEL_NAME = os.path.basename(ADAPTER.rstrip("/"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "model": ADAPTER, "base": BASE, "device": DEVICE})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/decide", "/v1/systemone"):
            return self._send(404, {"error": "not found"})
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            start = time.perf_counter()
            with LOCK, torch.inference_mode():  # one decision at a time on the GPU
                response = DECIDE(request["state"], request["questions"])
            response["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            response["model"] = MODEL_NAME  # ours, not the base model's name
            self._send(200, response)
        except Exception as error:  # report, never crash the server
            self._send(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    port = int(os.environ.get("DECIDER_PORT", "8765"))
    print(f"loading {ADAPTER} on {DEVICE} ...", flush=True)
    DECIDE = load()
    # Warm up so the first real decision does not pay for kernel selection.
    DECIDE("warm up", {"q": {"type": "noul", "instructions": "Is this a warm-up?"}})
    print(f"serving on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
