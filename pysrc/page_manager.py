import contextlib
from typing import Callable, Dict, Optional
import torch

class LazyModule(torch.nn.Module):
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

def build_lazy_page_table(factories: Dict[str, Callable[[], torch.nn.Module]], onload_device: torch.device, offload_device: torch.device, keep_in_gpu: float = 0.0) -> Dict[str, LazyModule]:
    return {name: LazyModule(factory, onload_device, offload_device, keep_in_gpu) for name, factory in factories.items()}
