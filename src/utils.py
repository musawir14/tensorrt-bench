import time, statistics as stats, torch

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_inference(step_fn, warmup=20, iters=100):
    for _ in range(warmup):
        step_fn()
    cuda_sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step_fn()
        cuda_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    times_sorted = sorted(times)
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    return {"p50_ms": stats.median(times), "p95_ms": p95, "mean_ms": sum(times)/len(times), "n": len(times)}
