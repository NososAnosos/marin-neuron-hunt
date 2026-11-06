"""The GPU side: one Modal H200 container loads Marin 32B once and runs the day's requests in order.

Weights and atlas stats come from the `marin-neuron-atlas` volume (written by the atlas pipeline).
"""

import time
import traceback

import modal

VOL = "/vol"
MODEL_DIR = f"{VOL}/models/marin-32b-bf16"
POLE_STATS = f"{VOL}/runs/stageB2/summaries/pole_stats.parquet"

app = modal.App("marin-hunt-runner")
vol = modal.Volume.from_name("marin-neuron-atlas")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.*", "transformers==5.10.2", "accelerate", "safetensors", "pyarrow", "numpy", "pyyaml")
    .add_local_python_source("runner")
)


def run_one(m, item):
    from runner.experiments import RUNNERS
    from runner.prompts import check

    t0 = time.time()
    problems = check(m.tok, item["request"])
    if problems:
        res = dict(outcome="inconclusive", reason="rejected: " + "; ".join(problems), summary={}, per_prompt=[])
    else:
        try:
            res = RUNNERS[item["request"]["type"]](m, item["request"], item["id"])
        except Exception:
            res = dict(outcome="inconclusive", reason="runner error: " + traceback.format_exc(limit=3)[-1500:],
                       summary={}, per_prompt=[])
    res["seconds"] = time.time() - t0
    return res


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=6 * 3600)
def run_batch(items):
    import torch

    from runner.model import Marin

    m = Marin.load(MODEL_DIR, POLE_STATS)
    gpu = torch.cuda.get_device_name()
    out = []
    for item in items:
        res = run_one(m, item)
        res["gpu"] = gpu
        print(f"{item['id']}: {res['outcome']} ({res['seconds']:.0f}s)", flush=True)
        out.append(res)
    return out
