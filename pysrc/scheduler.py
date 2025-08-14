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
    def module(self, name: str, *args):
        entry = self.mm._tiers[name]
        order = sorted(entry['offsets'].items(), key=lambda kv: kv[1])
        file_offsets = [off for _, off in order]
        sizes = [entry['sizes'][k] for k, _ in order]
        _pager.model_set_weights_layout(file_offsets, sizes)
        total_bytes = _pager.model_planned_bytes() if hasattr(_pager, 'model_planned_bytes') else sum(entry['sizes'].values())
        _pager.model_reserve(int(total_bytes))
        if hasattr(_pager, 'model_stage_file'):
            _pager.model_stage_file(entry['path'], int(8 << 20))
        _pager.model_prefetch()
        try:
            # Call the registered Python kernel through C extension
            result = _pager.launch_kernel(name, list(args))
            yield result
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
