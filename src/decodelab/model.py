"""A small decoder with explicit shapes and an ordinary PyTorch attention path.

These seeded, untrained weights are a numerical fixture, not a language model
checkpoint. The architecture uses learned positions and standard multi-head
attention, deliberately keeping RoPE/GQA/fused kernels out of the first lesson.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .cache import KVCache


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 512

    def __post_init__(self):
        if any(
            value <= 0
            for value in (self.vocab_size, self.dim, self.n_heads, self.n_layers, self.max_seq_len)
        ):
            raise ValueError("all model dimensions must be positive")
        if self.dim % self.n_heads:
            raise ValueError("model dim must be divisible by n_heads")


def record(trace: list | None, name: str, tensor: torch.Tensor):
    if trace is not None:
        trace.append({"name": name, "shape": list(tensor.shape)})


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer: int):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.layer = layer
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.output = nn.Linear(config.dim, config.dim)

    def forward(self, x, cache=None, past_length=0, trace=None):
        batch, length, width = x.shape  # [B, T, D]
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        record(trace, f"layer{self.layer}.query", query)
        if cache is not None:
            key, value = cache.update(self.layer, key, value)
        record(trace, f"layer{self.layer}.key_with_history", key)

        # [B, H, T_new, T_past + T_new]. For a chunk, offset the causal diagonal.
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_length + torch.arange(length, device=x.device)
        key_positions = torch.arange(key.shape[-2], device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        record(trace, f"layer{self.layer}.attention_scores", scores)
        attended = (probabilities @ value).transpose(1, 2).contiguous()
        return self.output(attended.reshape(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attention = CausalAttention(config, layer)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim),
        )

    def forward(self, x, cache, past_length, trace):
        x = x + self.attention(self.norm1(x), cache, past_length, trace)
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(DecoderBlock(config, layer) for layer in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens, cache: KVCache | None = None, trace: list | None = None):
        if tokens.ndim != 2 or 0 in tokens.shape:
            raise ValueError("tokens must be a non-empty [batch, sequence] tensor")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must use torch.long integer IDs")
        if tokens.device != self.token_embedding.weight.device:
            raise ValueError("tokens and model must be on the same device")
        if tokens.min().item() < 0 or tokens.max().item() >= self.config.vocab_size:
            raise ValueError("token IDs must be within the model vocab")
        past_length = cache.length if cache is not None else 0
        if past_length + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("model context length exceeded")
        positions = past_length + torch.arange(tokens.shape[1], device=tokens.device)
        record(trace, "input_ids", tokens)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        record(trace, "embedding", x)
        for block in self.blocks:
            x = block(x, cache, past_length, trace)
        record(trace, "hidden_states", x)
        logits = self.lm_head(self.final_norm(x))
        record(trace, "logits", logits)
        return logits
