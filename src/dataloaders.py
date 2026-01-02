import torch
def random_batch(batch_size=32, shape=(3,224,224), device="cuda", dtype=torch.float32):
    return torch.randn(batch_size, *shape, device=device, dtype=dtype)
