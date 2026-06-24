"""Shared helpers for per-slide ground-truth caching in training losses."""

from __future__ import annotations

from typing import Optional

import torch


def gt_batch_cache_key(cell_id: Optional[torch.Tensor]) -> Optional[bytes]:
    """Stable cache key for one batched slide (dense batch row order)."""
    if cell_id is None:
        return None
    return cell_id.detach().to(torch.int64).cpu().contiguous().numpy().tobytes()


def gt_row_cache_key(
    cell_id: Optional[torch.Tensor],
    batch_idx: int,
) -> Optional[bytes]:
    """Cache key for one row of a dense batch (``cell_id[batch_idx]``)."""
    if cell_id is None:
        return None
    if cell_id.dim() >= 2:
        row = cell_id[batch_idx]
    else:
        row = cell_id
    return row.detach().to(torch.int64).cpu().contiguous().numpy().tobytes()


def cache_key_matches(
    cell_id: Optional[torch.Tensor],
    cached_cell_id: Optional[torch.Tensor],
) -> bool:
    """Return True when ``cached_cell_id`` matches the current batch."""
    if cell_id is None or cached_cell_id is None:
        return False
    if cell_id.shape != cached_cell_id.shape:
        return False
    return torch.equal(cell_id.detach(), cached_cell_id)


def cache_key_seed(cache_key: Optional[bytes], epoch: int) -> int:
    """Deterministic RNG seed from epoch + slide identity."""
    base = int(epoch) * 1_000_003
    if cache_key is None:
        return base % (2**31 - 1)
    return (base + hash(cache_key)) % (2**31 - 1)
