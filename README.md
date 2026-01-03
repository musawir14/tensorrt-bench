# PyTorch → TensorRT Mini Benchmark (ResNet-50)

Reproducible benchmark that measures **inference latency** for ResNet-50 across:
- **PyTorch FP32**
- **PyTorch FP16**
- **Torch-TensorRT FP16** (optional)

It reports **p50 / p95 / mean** latency per batch using CUDA synchronization and warm-ups.

---

## Requirements
- NVIDIA GPU with recent driver (CUDA 12.x runtime compatible)
- Linux (e.g., Ubuntu 22.04)
- Python 3.10+
- PyTorch 2.4.x, TorchVision 0.19.x, Torch-TensorRT 2.4.x

---

## Quick Start (one screen)

```bash
# create project
mkdir -p ~/tensorrt-bench/src && cd ~/tensorrt-bench
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# deps
cat > requirements.txt <<'TXT'
torch==2.4.1
torchvision==0.19.1
torch_tensorrt==2.4.0
numpy
TXT
pip install -r requirements.txt

# code
cat > src/models.py <<'PY'
import torchvision.models as models
def load_resnet50(device="cuda"):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    m.eval()
    return m
PY

cat > src/utils.py <<'PY'
import time, statistics as stats, torch
def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def time_inference(step, warmup=20, iters=100):
    for _ in range(warmup): step(); _sync()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); step(); _sync()
        ts.append((time.perf_counter()-t0)*1e3)
    ts.sort()
    return {"p50_ms": ts[len(ts)//2],
            "p95_ms": ts[int(0.95*(len(ts)-1))],
            "mean_ms": sum(ts)/len(ts),
            "n": len(ts)}
PY

cat > src/dataloaders.py <<'PY'
import torch
def random_batch(b=32, shape=(3,224,224), device="cuda", dtype=torch.float32):
    return torch.randn(b, *shape, device=device, dtype=dtype)
PY

cat > src/bench.py <<'PY'
import argparse, torch
from models import load_resnet50
from dataloaders import random_batch
from utils import time_inference

def bench_pytorch(precision="fp32", batch=32, iters=50):
    model = load_resnet50("cuda")
    if precision=="fp16": model = model.half(); dtype=torch.float16
    else: dtype=torch.float32
    @torch.no_grad()
    def step(): _ = model(random_batch(batch, dtype=dtype))
    return time_inference(step, warmup=20, iters=iters)

def bench_trt(precision="fp16", batch=32, iters=50):
    import torch_tensorrt as trt
    model = load_resnet50("cuda")
    enabled = {torch.float16} if precision=="fp16" else {torch.float32}
    trt_model = trt.compile(model,
                            inputs=[trt.Input((batch,3,224,224))],
                            enabled_precisions=enabled,
                            require_full_compilation=False)
    dtype = torch.float16 if precision=="fp16" else torch.float32
    @torch.no_grad()
    def step(): _ = trt_model(random_batch(batch, dtype=dtype))
    return time_inference(step, warmup=20, iters=iters)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pytorch","trt"], default="pytorch")
    ap.add_argument("--precision", choices=["fp32","fp16"], default="fp32")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    out = bench_pytorch(a.precision,a.batch,a.iters) if a.mode=="pytorch" else bench_trt(a.precision,a.batch,a.iters)
    print({"mode":a.mode,"precision":a.precision,"batch":a.batch,**out})
PY

printf ".venv/\n__pycache__/\n*.pyc\n*.engine\n" > .gitignore
```
## Environment

```bash
# GPU/driver
nvidia-smi

# Python libs
python - << 'PY'
import torch, torchvision
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
try:
    import torch_tensorrt as trt
    print("torch_tensorrt:", trt.__version__)
except Exception as e:
    print("torch_tensorrt not loaded:", e)
PY
```
- **GPU:** NVIDIA RTX 4000 SFF Ada (20 GB)
- **Driver:** 560.35.05
- **CUDA (driver):** 12.6
- **PyTorch:** 2.4.1 (+cu124)
- **TorchVision:** 0.19.1
- **Torch-TensorRT:** tbd

## Results — Latency (ms), batch=32, iters=30

| Mode    | Precision | Batch | p50 (ms) | p95 (ms) | mean (ms) |
|---------|-----------|------:|---------:|---------:|----------:|
| PyTorch | FP32      |   32  |  44.076  |  44.108  |   44.076  |
| PyTorch | FP16      |   32  |  21.758  |  21.816  |   21.767  |

