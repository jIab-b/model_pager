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
import contextlib
import torch
from typing import Dict, Callable, Optional
from .file_indexing import index_weights_any

# ------------------------------------------------------------------ #
# Lightweight LazyModule (was in page_manager.py)                    #
# ------------------------------------------------------------------ #
class LazyModule(torch.nn.Module):
    """Wrap a factory so weights load on first use and offload after call."""
    def __init__(self, factory: Callable[[], torch.nn.Module], onload_device: torch.device, offload_device: torch.device, keep_in_gpu: float = 0.0):
        super().__init__()
        self._factory = factory
        self._onload = onload_device
        self._offload = offload_device
        self._keep = keep_in_gpu
        self._module: Optional[torch.nn.Module] = None
    def _ensure_loaded(self):
        if self._module is None:
            self._module = self._factory().to(self._onload)
            self._module.eval()
    @contextlib.contextmanager
    def _loaded(self):
        self._ensure_loaded()
        try:
            yield self._module
        finally:
            if self._keep == 0:
                self._module.to(self._offload)
                torch.cuda.empty_cache()
    def __getattr__(self, item):
        if item in self.__dict__:
            return super().__getattribute__(item)
        self._ensure_loaded()
        return getattr(self._module, item)
    def forward(self, *args, **kwargs):
        with self._loaded() as mod:
            return mod(*args, **kwargs)

def build_lazy_page_table(factories: Dict[str, Callable[[], torch.nn.Module]], onload_device: torch.device, offload_device: torch.device, keep_in_gpu: float = 0.0) -> Dict[str, 'LazyModule']:
    """Return {name: LazyModule} for each factory."""
    return {name: LazyModule(f, onload_device, offload_device, keep_in_gpu) for name, f in factories.items()}

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
        self.shapes = {k: v.shape for k, v in self._skeleton.named_parameters()}
    def materialise(self, state_dict: Dict[str, torch.Tensor], device: torch.device):
        """Return a *real* module with weights loaded on `device`."""
        mod = self._skeleton.__class__.__new__(self._skeleton.__class__)
        mod.__dict__ = self._skeleton.__dict__.copy()
        mod.to(device, dtype=next(iter(state_dict.values())).dtype)
        with torch.no_grad():
            for name, param in mod.named_parameters(recurse=True):
                v_src = state_dict[name]
                if v_src.device == device:
                    v_cuda = v_src
                else:
                    v_cuda = v_src.to(device, non_blocking=True)
                setattr(mod, name, torch.nn.Parameter(v_cuda, requires_grad=False))
            for name, buf in mod.named_buffers(recurse=True):
                if name in state_dict:
                    setattr(mod, name, state_dict[name].to(device, non_blocking=True))
        mod.eval()
        return mod

class ModuleDescriptor:
    def __init__(self, shapes: Dict[str, torch.Size]):
        self.shapes = shapes
    def materialise(self, state_dict: Dict[str, torch.Tensor], device):
        raise NotImplementedError

class DeviceBackend:
    def __init__(self, device: str | None = None):
        self.device = device
    def synchronize(self):
        pass
    def memory_allocated(self) -> int:
        return 0

class TorchCudaBackend(DeviceBackend):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.torch_device = torch.device(device)
    def synchronize(self):
        torch.cuda.synchronize()
    def memory_allocated(self) -> int:
        return torch.cuda.memory_allocated(self.torch_device)

# ------------------------------------------------------------------ #
# MemoryManager                                                      #
# ------------------------------------------------------------------ #
class MemoryManager:
    def __init__(self, gpu: str = "cuda", vram_limit_mb: int | None = None, device_backend: DeviceBackend | None = None):
        self.backend = device_backend if device_backend is not None else TorchCudaBackend(gpu)
        self.vram_limit = None if vram_limit_mb is None else vram_limit_mb * 1024**2
        self._tiers: Dict[str, Dict] = {}
        self._page_table: Dict[str, Dict] = {}
    def _evict_until(self, bytes_needed: int):
        """Release least-recently-used modules until freeing >= bytes_needed."""
        freed = 0
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
        self.backend.synchronize()
        used = self.backend.memory_allocated()
        if used + extra > self.vram_limit:
            need = used + extra - self.vram_limit
            freed = self._evict_until(need)
            self.backend.synchronize()
            used2 = self.backend.memory_allocated()
            if used2 + extra > self.vram_limit:
                raise RuntimeError("VRAM cap exceeded even after eviction")
    def register(self, name: str, meta: MetaModule, weights_path: str):
        idx = index_weights_any(weights_path)
        self._tiers[name] = dict(
            meta=meta,
            path=idx["path"],
            files=idx["files"],
            tensor_files=idx["tensor_files"],
            header_lens=idx["header_lens"],
            is_sharded=idx["is_sharded"],
            gpu=None,
            cpu=None,
            t=0.0,
            um_ptr=None,
            um_pages=0,
            offsets=idx["offsets"],
            sizes=idx["sizes"],
            dtypes=idx["dtypes"],
            shapes=idx["shapes"],
        )
        for tname, nbytes in idx["sizes"].items():
            self._page_table[f"{name}.{tname}"] = dict(
                path=self._tiers[name]["tensor_files"][tname],
                bytes=nbytes,
                offset=idx["offsets"][tname],
                module=name,
                dtype=idx["dtypes"][tname],
            )
    def _load_from_disk(self, name):
        raise NotImplementedError("Weight loading disabled in Python-only index mode")
    def acquire(self, name: str):
        return self._tiers[name]["gpu"]
    def release(self, name: str):
        e = self._tiers[name]
        e["gpu"] = None
    @contextlib.contextmanager
    def use(self, name: str):
        yield self.acquire(name)