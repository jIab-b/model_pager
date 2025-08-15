from __future__ import annotations
from contextlib import contextmanager
import torch
import page_table_ext as _pager

from .index import index_safetensors_any
from .registry import PagerRegistry
from .state import set_state, get_state


@contextmanager
def pager_context(weights_path: str, device_id: int = 0, compute_streams: int = 2, io_streams: int = 2):
    _pager.runtime_init(device_id, compute_streams, io_streams)
    meta = index_safetensors_any(weights_path)
    reg = PagerRegistry(meta)
    reg.register_all()
    device = torch.device(f"cuda:{device_id}")
    set_state({
        "device": device,
        "meta": meta,
        "lut": reg.name_to_loc,
        "registry": reg,
    })
    try:
        yield get_state()
    finally:
        _pager.wait_all()
        reg.close_all()
        set_state(None)
        _pager.runtime_shutdown()


