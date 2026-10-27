import pytest
import torch

from decodelab.cache import KVCache
from decodelab.generation import generate
from decodelab.model import ModelConfig, TinyDecoder


@pytest.fixture
def model():
    torch.manual_seed(11)
    return TinyDecoder(
        ModelConfig(vocab_size=32, dim=24, n_heads=3, n_layers=2, max_seq_len=24)
    ).eval()


@pytest.mark.parametrize("mode", ["dynamic", "static"])
@pytest.mark.parametrize("chunks", [(4, 1, 1), (2, 3, 1), (1, 1, 4)])
def test_cached_logits_match_full_sequence_for_chunked_inputs(model, mode, chunks):
    # Catches wrong positional offsets, per-layer caches and cached causal masks.
    tokens = torch.tensor([[1, 7, 3, 9, 2, 5], [8, 6, 4, 2, 1, 3]])
    cache = KVCache(mode, capacity=24)
    pieces = []
    offset = 0
    with torch.inference_mode():
        expected = model(tokens)
        for size in chunks:
            pieces.append(model(tokens[:, offset : offset + size], cache=cache))
            offset += size
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-5, atol=1e-6)
    assert cache.length == 6


def test_attention_does_not_read_future_tokens(model):
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 9, 8]]))
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)


def test_all_cache_modes_generate_identical_greedy_tokens(model):
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    results = [
        generate(model, prompt, max_new_tokens=6, cache_mode=mode)
        for mode in ("none", "dynamic", "static")
    ]
    for result in results:
        assert result.tokens.shape == (2, 9)
        assert torch.equal(result.tokens[:, :3], prompt)
        assert len(result.token_times_s) == 6
        assert result.total_s >= result.token_times_s[-1] > 0
    assert torch.equal(results[0].tokens, results[1].tokens)
    assert torch.equal(results[0].tokens, results[2].tokens)


def test_static_cache_reuses_storage_and_returns_only_written_tokens():
    cache = KVCache("static", capacity=8)
    key = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    cached_key, _ = cache.update(0, key, key + 10)
    pointer = cached_key.data_ptr()
    next_key = torch.full((1, 2, 1, 3), 99.0)
    updated_key, updated_value = cache.update(0, next_key, next_key + 10)
    assert updated_key.data_ptr() == pointer
    assert updated_key.shape == (1, 2, 3, 3)
    torch.testing.assert_close(updated_key[:, :, :2], key)
    assert updated_value[0, 0, 2, 0].item() == 109
    assert cache.used_bytes == 2 * 1 * 2 * 3 * 3 * 4
    assert cache.allocated_bytes == 2 * 1 * 2 * 8 * 3 * 4


def test_cache_overflow_fails_without_changing_existing_tokens():
    cache = KVCache("dynamic", capacity=2)
    key = torch.ones(1, 1, 2, 3)
    cache.update(0, key, key)
    with pytest.raises(ValueError, match="capacity"):
        cache.update(0, key[:, :, :1], key[:, :, :1])
    assert cache.length == 2


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_cache_rejects_batch_or_dtype_changes(mode):
    cache = KVCache(mode, capacity=8)
    key = torch.ones(1, 2, 1, 3)
    cache.update(0, key, key)
    with pytest.raises(ValueError, match="shape|dtype"):
        cache.update(0, key.expand(2, -1, -1, -1), key.expand(2, -1, -1, -1))
    with pytest.raises(ValueError, match="shape|dtype"):
        cache.update(0, key.double(), key.double())


def test_context_limit_and_input_validation(model):
    with pytest.raises(ValueError, match="context"):
        model(torch.ones(1, 25, dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.empty(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="long"):
        model(torch.ones(1, 2))
    with pytest.raises(ValueError, match="vocab"):
        model(torch.tensor([[32]]))
    with pytest.raises(ValueError, match="context"):
        generate(model, torch.tensor([[1, 2]]), max_new_tokens=23)
    with pytest.raises(ValueError, match="positive"):
        generate(model, torch.tensor([[1]]), max_new_tokens=0)


def test_configuration_rejects_incompatible_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(dim=25, n_heads=3)


def test_repeated_generation_starts_with_a_fresh_cache(model):
    prompt = torch.tensor([[1, 3, 5]])
    first = generate(model, prompt, max_new_tokens=3, cache_mode="static")
    second = generate(model, prompt, max_new_tokens=3, cache_mode="static")
    assert torch.equal(first.tokens, second.tokens)
