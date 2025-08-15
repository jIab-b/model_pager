from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import page_table_ext as _pager


class PagerRegistry:
    def __init__(self, meta: Dict):
        self.meta = meta
        self.handles: Dict[str, int] = {}
        self.name_to_loc: Dict[str, Tuple[int, int]] = {}

    def register_all(self):
        files = sorted(set(self.meta["tensor_files"].values()))
        for f in files:
            names = [n for n, tf in self.meta["tensor_files"].items() if tf == f]
            names.sort(key=lambda n: self.meta["offsets"][n])
            base = 8 + self.meta["header_lens"][f]
            file_offsets = [base + self.meta["offsets"][n] for n in names]
            sizes = [self.meta["sizes"][n] for n in names]
            h = _pager.model_register(f, file_offsets, sizes)
            self.handles[f] = h
            for i, n in enumerate(names):
                self.name_to_loc[n] = (h, i)

    def close_all(self):
        for f, h in self.handles.items():
            _pager.model_close(h)
        self.handles.clear()
        self.name_to_loc.clear()


