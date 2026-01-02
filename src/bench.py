import argparse, torch
from models import load_resnet50
from dataloaders import random_batch
from utils import time_inference

def bench_pytorch(precision="fp32", batch=32, iters=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_resnet50(device)
    dtype = torch.float16 if precision=="fp16" else torch.float32
    if precision=="fp16":
        model = model.half()

    @torch.no_grad()
    def step():
        x = random_batch(batch, device=device, dtype=dtype)
        _ = model(x)
    return time_inference(step, warmup=20, iters=iters)

def bench_trt(precision="fp16", batch=32, iters=50):
    import torch_tensorrt as trt
    device = "cuda"
    model = load_resnet50(device)
    enabled = {torch.float16} if precision=="fp16" else {torch.float32}
    trt_model = trt.compile(model,
        inputs=[trt.Input((batch,3,224,224))],
        enabled_precisions=enabled,
        require_full_compilation=False)
    dtype = torch.float16 if precision=="fp16" else torch.float32

    @torch.no_grad()
    def step():
        x = random_batch(batch, device=device, dtype=dtype)
        _ = trt_model(x)
    return time_inference(step, warmup=20, iters=iters)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pytorch","trt"], default="pytorch")
    ap.add_argument("--precision", choices=["fp32","fp16"], default="fp32")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if args.mode=="pytorch":
        out = bench_pytorch(args.precision, args.batch, args.iters)
        print({"mode":"pytorch","precision":args.precision,"batch":args.batch,**out})
    else:
        out = bench_trt(args.precision, args.batch, args.iters)
        print({"mode":"torchtrt","precision":args.precision,"batch":args.batch,**out})

if __name__ == "__main__":
    main()
