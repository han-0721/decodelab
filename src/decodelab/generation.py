"""One greedy generation loop shared by demo, checks and benchmarks."""

from dataclasses import dataclass
from time import perf_counter

import torch

from .cache import KVCache
from .model import TinyDecoder


@dataclass
class GenerationResult:
    tokens: torch.Tensor
    token_times_s: list[float]  # cumulative, measured after each selected token is ready
    total_s: float
    cache_used_bytes: int
    cache_allocated_bytes: int
    cuda_peak_allocated_bytes: int | None
    cuda_baseline_allocated_bytes: int | None


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(
    model: TinyDecoder, prompt: torch.Tensor, max_new_tokens: int = 16, cache_mode: str = "dynamic"
) -> GenerationResult:
    if cache_mode not in ("none", "dynamic", "static"):
        raise ValueError("cache_mode must be none, dynamic or static")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt.ndim != 2 or 0 in prompt.shape:
        raise ValueError("prompt must be a non-empty [batch, sequence] tensor")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generated tokens exceeds model context")

    device = prompt.device
    synchronize(device)
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    tokens = prompt.clone()
    # Static allocation reserves the declared maximum request length.
    capacity = prompt.shape[1] + max_new_tokens
    cache = None if cache_mode == "none" else KVCache(cache_mode, capacity)
    token_times = []
    for step in range(max_new_tokens):
        # First call consumes the full prompt; cached calls then consume one new token.
        inputs = tokens if cache is None or step == 0 else tokens[:, -1:]
        logits = model(inputs, cache=cache)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        synchronize(device)
        token_times.append(perf_counter() - start)
        del logits
    synchronize(device)
    total = perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return GenerationResult(
        tokens,
        token_times,
        total,
        cache.used_bytes if cache is not None else 0,
        cache.allocated_bytes if cache is not None else 0,
        peak,
        baseline,
    )
