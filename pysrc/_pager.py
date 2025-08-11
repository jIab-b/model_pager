from __future__ import annotations
import os, pathlib, torch
from torch.utils.cpp_extension import load

_ext_path = pathlib.Path(__file__).resolve().parent.parent / "csrc"

# Build the extension lazily and cache under torch extensions dir
pager = load(name="weight_pager_ext",
            sources=[str(_ext_path / "weight_pager.cu"), str(_ext_path / "um_tensor.cpp")],
            extra_include_paths=[str(_ext_path)],
            verbose=False)

PAGE_BYTES = 64 * 1024  # must match C++

reserve = pager.pager_reserve
prefetch = pager.pager_prefetch
evict    = pager.pager_evict
