"""Two exact KV storage policies; neither drops or quantizes attention history."""

import torch


class KVCache:
    """Store each layer's K/V as [batch, heads, sequence, head_dim].

    dynamic: concatenate valid tokens on every update.
    static: allocate capacity once and write into a slice.
    One cache belongs to one model's generation request/batch.
    """

    def __init__(self, mode: str, capacity: int):
        if mode not in ("dynamic", "static"):
            raise ValueError("cache mode must be dynamic or static")
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.mode = mode
        self.capacity = capacity
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lengths: dict[int, int] = {}

    @property
    def length(self) -> int:
        return self._lengths.get(0, 0)

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.shape[-2] == 0
            or key.dtype != value.dtype
            or key.device != value.device
        ):
            raise ValueError("K/V must have matching non-empty 4D shape, dtype and device")
        previous_length = self._lengths.get(layer, 0)
        end = previous_length + key.shape[-2]
        if end > self.capacity:
            raise ValueError(f"cache capacity {self.capacity} exceeded by {end} tokens")

        if layer in self._buffers:
            old_key, old_value = self._buffers[layer]
            expected = (old_key.shape[0], old_key.shape[1], old_key.shape[3])
            incoming = (key.shape[0], key.shape[1], key.shape[3])
            if expected != incoming or key.dtype != old_key.dtype or key.device != old_key.device:
                raise ValueError("cache update changed batch/head shape, dtype or device")
        elif self.mode == "static":
            shape = (*key.shape[:2], self.capacity, key.shape[-1])
            old_key = key.new_empty(shape)
            old_value = value.new_empty(shape)

        if self.mode == "dynamic":
            if previous_length:
                stored_key = torch.cat((old_key, key), dim=-2)
                stored_value = torch.cat((old_value, value), dim=-2)
            else:
                # Own storage, rather than retaining a larger QKV projection view.
                stored_key, stored_value = key.clone(), value.clone()
        else:
            old_key[:, :, previous_length:end].copy_(key)
            old_value[:, :, previous_length:end].copy_(value)
            stored_key, stored_value = old_key, old_value

        self._buffers[layer] = (stored_key, stored_value)
        self._lengths[layer] = end
        # Never expose uninitialized slots in the preallocated buffer.
        return stored_key[:, :, :end], stored_value[:, :, :end]

    @property
    def used_bytes(self) -> int:
        return sum(
            2
            * key.shape[0]
            * key.shape[1]
            * self._lengths[layer]
            * key.shape[3]
            * key.element_size()
            for layer, (key, _) in self._buffers.items()
        )

    @property
    def allocated_bytes(self) -> int:
        """Logical cache tensor storage, excluding allocator/transient overhead."""
        return sum(
            (key.numel() + value.numel()) * key.element_size()
            for key, value in self._buffers.values()
        )
