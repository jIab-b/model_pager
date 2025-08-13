from __future__ import annotations
import contextlib
from typing import Iterable, Callable, Dict, Any
import page_table_ext as _pager

class SequentialScheduler:
    """Simplest possible scheduler: acquires one module at a time.

    It is meant for very-low-VRAM scenarios: a module is loaded, the caller
    executes its compute function, and the weights are released immediately
    after.
    """

    def __init__(self, memory_manager):
        self.mm = memory_manager

    @contextlib.contextmanager
    def module(self, name: str):
        # Reserve and prefetch full model memory
        entry = self.mm._tiers[name]
        total_bytes = sum(entry["sizes"].values())
        _pager.model_reserve(total_bytes)
        _pager.model_prefetch()
        try:
            with self.mm.use(name) as mod:
                yield mod
        finally:
            _pager.model_evict()

    def run(self, order: Iterable[str], handlers: Dict[str, Callable[[Any], None]]):
        """Execute `handlers[name](module)` for every name in `order`.

        Args:
            order: iterable of module names (execution sequence).
            handlers: mapping name→callable that consumes the materialised module.
        """
        for name in order:
            fn = handlers.get(name)
            if fn is None:
                raise KeyError(f"No handler registered for module '{name}'")
            with self.module(name) as mod:
                fn(mod)

    def update_schedule(self, module_ids: Iterable[str], priorities: Iterable[int]) -> None:
        _pager.update_schedule(module_ids, priorities)

    def process_schedule(self) -> None:
        _pager.process_schedule()
