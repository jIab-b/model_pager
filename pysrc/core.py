# SPDX-License-Identifier: Apache-2.0
"""
Generic 3-tier memory manager for torch models built on the **meta**
device.  No parameter tensor lives outside our control.

Tier-0  (Disk)         : .safetensors / pt files
Tier-1* (CPU-pinned)    : optional staging buffer (can be size-capped)
Tier-2  (GPU)          : the active layer
(*) Tier-1 can be bypassed to stream directly Disk→GPU if you prefer.
"""

from __future__ import annotations
import mmap, os, contextlib, functools, pathlib, time
from typing import Dict, Callable, Optional

import safetensors.torch as safe
import torch
from . import _pager
from . import _alloc

__all__ = ["MemoryManager", "MetaModule", "register_kernel"]

# ------------------------------------------------------------------ #
# Optional user kernels registry                                     #
# ------------------------------------------------------------------ #
_KERNELS: Dict[str, Callable] = {}

def register_kernel(name: str, fn: Callable):
    _KERNELS[name] = fn


# ------------------------------------------------------------------ #
# MetaModule – instantiate nn.Module on meta device w/out weights    #
# ------------------------------------------------------------------ #
class MetaModule(torch.nn.Module):
    """
    Build the module architecture with **no real data**.
    `builder` must accept a `device` kwarg.
    """
    def __init__(self, builder: Callable[[], torch.nn.Module]):
        super().__init__()
        self._skeleton = builder().to(torch.device("meta"))
        # keep a param-name → torch.Size dict for sanity checks
        self.shapes = {k: v.shape for k, v in self._skeleton.named_parameters()}

    def materialise(self, state_dict: Dict[str, torch.Tensor], device: torch.device):
        """Return a *real* module with weights loaded on `device`."""
        mod = self._skeleton.__class__.__new__(self._skeleton.__class__)
        mod.__dict__ = self._skeleton.__dict__.copy()
        mod.to(device, dtype=next(iter(state_dict.values())).dtype)

        with torch.no_grad():
            for name, param in mod.named_parameters(recurse=True):
                v = state_dict[name].to(device, non_blocking=True)
                setattr(mod, name, torch.nn.Parameter(v, requires_grad=False))
            for name, buf in mod.named_buffers(recurse=True):
                if name in state_dict:
                    setattr(mod, name, state_dict[name].to(device, non_blocking=True))
        mod.eval()
        return mod


# ------------------------------------------------------------------ #
# MemoryManager                                                      #
# ------------------------------------------------------------------ #
class MemoryManager:
    def __init__(self, gpu: str = "cuda", pinned_limit_mb: int = 512, vram_limit_mb: int | None = None):
        self.gpu = torch.device(gpu)
        self.pinned_limit = pinned_limit_mb * 1024**2
        self.vram_limit = None if vram_limit_mb is None else vram_limit_mb * 1024**2

        if self.vram_limit is not None:
            _alloc.set_cap(self.vram_limit)

        self._tiers: Dict[str, Dict] = {}      # name → {meta, path, …}

        # size-tracked FIFO pinned cache
        self._pinned_used = 0
        self._pinned_fifo: Dict[str, Dict[str, torch.Tensor]] = {}

    # --------------- internal -------------------------------------- #
    def _evict_until(self, bytes_needed: int):
        """Release least-recently-used modules until freeing >= bytes_needed."""
        freed = 0
        # sort tiers by last used timestamp ascending
        for name, info in sorted(self._tiers.items(), key=lambda kv: kv[1]["t"]):
            if info["gpu"] is None:
                continue
            size = sum(p.element_size() * p.nelement() for p in info["gpu"].parameters())
            self.release(name)
            freed += size
            if freed >= bytes_needed:
                break
        return freed

    def _ensure_vram(self, extra: int):
        if self.vram_limit is None:
            return
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated(self.gpu)
        if used + extra > self.vram_limit:
            need = used + extra - self.vram_limit
            freed = self._evict_until(need)
            torch.cuda.synchronize()
            used2 = torch.cuda.memory_allocated(self.gpu)
            if used2 + extra > self.vram_limit:
                raise RuntimeError("VRAM cap exceeded even after eviction")

    # ---------------- registration -------------------------------- #
    def register(self, name: str, meta: MetaModule, weights_path: str):
        self._tiers[name] = dict(meta=meta, path=weights_path,
                                 cpu=None, gpu=None, t=0.0,
                                 um_ptr=None, um_pages=0)

    # ---------------- helpers ------------------------------------- #
    def _load_from_disk(self, name):
        entry = self._tiers[name]
        if entry["cpu"] is not None:
            return entry["cpu"]             # cached in pinned

        obj = safe.load_file(entry["path"], device="cpu",  # still CPU mem
                             lowercase=False, framework="pt")
        # Allocate unified-memory pages once based on total size
        if entry["um_ptr"] is None:
            total_bytes = sum(v.element_size() * v.nelement() for v in obj.values())
            pages = (total_bytes + _pager.PAGE_BYTES - 1) // _pager.PAGE_BYTES
            entry["um_ptr"] = _pager.reserve(total_bytes)
            entry["um_pages"] = pages
            # TODO: copy individual param tensors into the UM block and record offsets
            entry["offsets"] = {}
        # move to pinned (async copy helper)
        for k, v in obj.items():
            obj[k] = v.pin_memory()
        size = sum(v.element_size() * v.nelement() for v in obj.values())
        # evict if needed
        while self._pinned_used + size > self.pinned_limit and self._pinned_fifo:
            evict_name, evict_sd = self._pinned_fifo.popitem(last=False)
            self._pinned_used -= sum(v.element_size() * v.nelement() for v in evict_sd.values())
            self._tiers[evict_name]["cpu"] = None
        self._pinned_fifo[name] = obj
        self._pinned_used += size
        entry["cpu"] = obj
        return obj

    # ---------------- public API ---------------------------------- #
    def acquire(self, name: str):
        """
        Materialise module `name` on GPU (tier-2) and return it.
        If a CUDA custom kernel is registered we will bind it.
        """
        e = self._tiers[name]
        if e["gpu"] is None:
            state = self._load_from_disk(name)
            # pre-check VRAM use
            size = sum(v.element_size() * v.nelement() for v in state.values())
            self._ensure_vram(size)
            # prefetch weights into GPU memory (UMA)
            if e["um_ptr"] is not None and e["um_pages"]:
                _pager.prefetch(e["um_ptr"], e["um_pages"])

            e["gpu"] = e["meta"].materialise(state, self.gpu)
            e["t"] = time.time()
            # attach user kernel
            if name in _KERNELS:
                e["gpu"].forward = functools.partial(_KERNELS[name], e["gpu"])
        return e["gpu"]

    def release(self, name: str):
        "Drop GPU copy immediately (tier-2 → tier-1 or meta)."
        e = self._tiers[name]
        if e["gpu"] is not None:
            e["gpu"].to(torch.device("meta"))
            del e["gpu"]
            e["gpu"] = None
            torch.cuda.empty_cache()
            if e["um_ptr"] is not None and e["um_pages"]:
                _pager.evict(e["um_ptr"], e["um_pages"])

    # ---------- context manager for one-shot usage ---------------- #
    @contextlib.contextmanager
    def use(self, name: str):
        mod = self.acquire(name)
        try:
            yield mod
        finally:
            self.release(name)